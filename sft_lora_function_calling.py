#!/usr/bin/env python3
"""
🔥 HessGPT Mini 0.2B - SFT LoRA v9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ FIX v8: LoRALayer.forward cast float32 → évite bug dtype bfloat16
✅ FIX v8: lora_A / lora_B initialisés en float32 explicitement
✅ FIX v8: vérification requires_grad après setattr
✅ FIX v9: save toutes les 1h05 (au lieu de 50 min)
✅ FIX v9: merged_state construit en une seule passe propre (plus de doublons)
✅ FIX v9: validate_function_calling comptage correct (valid/total)
✅ FIX v9: IS_RESUME remis à False après le skip — évite de rester bloqué
✅ Sauvegarde : lora_checkpoint.pt + merged_checkpoint.pt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
import json
import re
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime

sys.path.append('./Core/Model')
from HessGpt import HessGPT

# ============================================
# CONFIG
# ============================================
SFT_DATA_DIR        = './sft_data'
PRETRAIN_CHECKPOINT = './tinyModel/hessgpt_mini_math_injected.pt'
OUTPUT_DIR          = './tinyModel/lora_adapters'
OUTPUT_CKPT         = f'{OUTPUT_DIR}/lora_checkpoint.pt'
OUTPUT_MERGED       = f'{OUTPUT_DIR}/merged_checkpoint.pt'

# ✅ FIX v9: 1h05 au lieu de 50 min
SAVE_EVERY_MINUTES  = 65
SAVE_INTERVAL_SEC   = SAVE_EVERY_MINUTES * 60   # 3900 secondes

EPOCHS          = 1
BATCH_SIZE      = 24
MAX_SEQ_LEN     = 2048
LORA_RANK       = 64
LEARNING_RATE   = 1e-4
GRAD_ACCUM      = 4

SPECIAL_TOKENS = {
    '<|system|>':       32000,
    '<|user|>':         32001,
    '<|assistant|>':    32002,
    '<|end|>':          32003,
    '<think>':          32004,
    '</think>':         32005,
    '<tool_call>':      32006,
    '</tool_call>':     32007,
    '<tool_response>':  32008,
    '</tool_response>': 32009,
    '<code>':           32010,
}

VOCAB_SIZE = 32011
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print("🔥 HessGPT Mini 0.2B — SFT LoRA v9")
print(f"   Input:    {PRETRAIN_CHECKPOINT}")
print(f"   Data:     {SFT_DATA_DIR}/")
print(f"   Output:   {OUTPUT_DIR}")
print(f"   Batch:    {BATCH_SIZE} × grad_accum {GRAD_ACCUM} = effective {BATCH_SIZE*GRAD_ACCUM}")
print(f"   LR:       {LEARNING_RATE:.0e}")
print(f"   Save:     toutes les {SAVE_EVERY_MINUTES} min")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# TOKENIZER
# ============================================
print(f"\n📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.add_special_tokens({
    'additional_special_tokens': list(SPECIAL_TOKENS.keys())
})
tokenizer.pad_token = tokenizer.eos_token
print(f"   ✅ Vocab: {len(tokenizer)} tokens")

# ============================================
# LOAD MODEL
# ============================================
print(f"\n🏗️  Loading pretrained model...")
checkpoint = torch.load(PRETRAIN_CHECKPOINT, map_location='cpu')
config     = checkpoint.get('config', {})

model = HessGPT(
    vocab_size         = VOCAB_SIZE,
    embed_dim          = config.get('embed_dim', 896),
    num_heads          = config.get('num_heads', 14),
    num_layers         = config.get('num_layers', 16),
    max_seq_len        = MAX_SEQ_LEN,
    dropout            = 0.0,
    use_rope           = True,
    use_yarn           = True,
    yarn_scale         = 2.0,
    yarn_original_max_len = 1024,
    use_swiglu         = True,
    n_kv_heads         = config.get('n_kv_heads', 7),
    use_qk_norm        = True,
    soft_cap           = 30.0,
    use_flash_attn     = True
)

state_dict = checkpoint['model_state_dict']
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=False)

# Poids de base → bf16 (frozen, économie mémoire)
model = model.to(device, dtype=torch.bfloat16)

pretrain_epoch = checkpoint.get('epoch', '?')
pretrain_loss  = checkpoint.get('last_loss', 0.0)
print(f"   ✅ Pretrain epoch={pretrain_epoch} | loss={pretrain_loss:.4f} | YaRN x2 (1024→2048) | bf16")

# ============================================
# LoRA — float32 explicite + cast dans forward
# ============================================
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, dropout):
        super().__init__()
        self.r       = r
        self.scaling = alpha / r
        # float32 explicite → gradients corrects même avec base bf16
        self.lora_A  = nn.Parameter(torch.zeros(in_features,  r, dtype=torch.float32))
        self.lora_B  = nn.Parameter(torch.zeros(r, out_features, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Cast en float32 pour éviter mismatch bf16 / float32
        x_f32    = x.float()
        lora_out = (self.dropout(x_f32) @ self.lora_A @ self.lora_B) * self.scaling
        return lora_out.to(x.dtype)   # recast vers le dtype d'entrée


class LinearWithLoRA(nn.Module):
    def __init__(self, base_layer, r, alpha, dropout):
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            r, alpha, dropout
        )

    def forward(self, x):
        return self.base_layer(x) + self.lora(x)


print(f"\n🔧 Applying LoRA (r={LORA_RANK}, alpha={LORA_RANK*2})...")

# Geler tous les poids du pretrain
for param in model.parameters():
    param.requires_grad = False

TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']
replaced = 0

for name, module in list(model.named_modules()):
    for target in TARGET_MODULES:
        if name.endswith(target) and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name  = name.split('.')[-1]
            parent      = model.get_submodule(parent_name) if parent_name else model
            lora_layer  = LinearWithLoRA(module, r=LORA_RANK, alpha=LORA_RANK*2, dropout=0.05)
            # Mettre SEULEMENT les poids LoRA sur device (en float32)
            lora_layer.lora = lora_layer.lora.to(device)
            setattr(parent, child_name, lora_layer)
            replaced += 1

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   ✅ {replaced} couches remplacées par LinearWithLoRA")
print(f"   ✅ Modules ciblés: {TARGET_MODULES}")
print(f"   📊 Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.1f}M ({100*trainable_params/total_params:.2f}%)")

# Vérification requires_grad
lora_grad_check = [(n, p.requires_grad, p.dtype) for n, p in model.named_parameters() if 'lora' in n]
if lora_grad_check:
    bad = [n for n, g, _ in lora_grad_check if not g]
    if bad:
        raise RuntimeError(f"❌ {len(bad)} poids LoRA ont requires_grad=False : {bad[:3]}")
    else:
        sample_info = [(n.split('.')[-1], str(d)) for n, _, d in lora_grad_check[:3]]
        print(f"   ✅ {len(lora_grad_check)} poids LoRA — requires_grad=True | dtypes: {sample_info}")
else:
    raise RuntimeError("❌ ERREUR CRITIQUE: aucun poids LoRA trouvé dans le modèle !")

if trainable_params == 0:
    raise RuntimeError("❌ Aucun paramètre entraînable — vérifier l'application du LoRA")

# ============================================
# DATASET
# ============================================
class SFTDataset(Dataset):
    def __init__(self, data_dir, max_seq_len):
        print(f"\n📦 Loading SFT dataset from {data_dir}/...")

        input_path   = os.path.join(data_dir, 'sft_input_ids.npy')
        labels_path  = os.path.join(data_dir, 'sft_labels.npy')
        lengths_path = os.path.join(data_dir, 'sft_lengths.npy')
        meta_path    = os.path.join(data_dir, 'sft_meta.json')

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"❌ {input_path} introuvable — lance prepare_sft_data.py d'abord")

        self.input_ids   = np.load(input_path)
        self.labels      = np.load(labels_path)
        self.lengths     = np.load(lengths_path)
        self.max_seq_len = max_seq_len

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"   ✅ {meta['n_samples']:,} samples | avg_len={meta['avg_seq_len']} tokens")
        else:
            print(f"   ✅ {len(self.input_ids):,} samples chargés")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        length    = min(int(self.lengths[idx]), self.max_seq_len)
        input_ids = torch.tensor(self.input_ids[idx, :length], dtype=torch.long)
        labels    = torch.tensor(self.labels[idx,    :length], dtype=torch.long)
        return input_ids, labels


def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels    = [item[1] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels    = torch.nn.utils.rnn.pad_sequence(labels,    batch_first=True, padding_value=-100)
    return input_ids, labels


# ============================================
# SCHEDULER — Cosine avec warmup
# ============================================
class CosineScheduler:
    def __init__(self, optimizer, max_lr, total_steps, warmup_ratio=0.03, min_lr_ratio=0.1):
        self.optimizer    = optimizer
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.current_step = 0

    def get_lr(self):
        step = self.current_step
        if step < self.warmup_steps:
            return self.max_lr * (step / max(self.warmup_steps, 1))
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def step(self):
        lr = self.get_lr()
        self.current_step += 1
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return [self.get_lr()]


# ============================================
# SAVE — lora_checkpoint + merged_checkpoint
# ============================================
def save_checkpoint(model, optimizer, scheduler, step, loss_val, batch_idx=-1):
    """
    Sauvegarde deux fichiers :
    1. lora_checkpoint.pt   → adaptateurs LoRA uniquement (pour resume)
    2. merged_checkpoint.pt → pretrain + LoRA fusionnés   (pour inférence)

    ✅ FIX v9: merged_state construit en une seule passe pour éviter les doublons.
    La fusion W_merged = W_base + (lora_A @ lora_B).T * scaling est faite en float32
    puis recastée en bfloat16 pour garder la cohérence avec le pretrain.
    """
    t0 = time.time()

    # ── 1. LoRA adapters uniquement ────────────────────────────
    lora_state = {
        name: param.data.cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    torch.save({
        'lora_state_dict': lora_state,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_step':  scheduler.current_step,
        'step':            step,
        'batch_idx':       batch_idx,
        'last_loss':       loss_val,
        'saved_at':        datetime.now().isoformat(),
        'config': {
            'lora_rank':      LORA_RANK,
            'lora_alpha':     LORA_RANK * 2,
            'target_modules': TARGET_MODULES,
            'max_seq_len':    MAX_SEQ_LEN,
            'yarn_scale':     2.0,
        },
        'special_tokens': SPECIAL_TOKENS,
    }, OUTPUT_CKPT)

    # ── 2. Merged checkpoint (pretrain + LoRA fusionnés) ───────
    # ✅ FIX v9: une seule passe sur named_parameters pour éviter les doublons.
    # On collecte d'abord les noms des couches LinearWithLoRA pour les fusionner,
    # puis on parcourt named_parameters une seule fois.

    # Carte : nom_module → LinearWithLoRA (pour fusion)
    lora_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LinearWithLoRA)
    }

    merged_state = {}

    for param_name, param in model.named_parameters():
        # Ignorer les poids LoRA (lora_A, lora_B) — ils seront fusionnés dans weight
        if '.lora.lora_A' in param_name or '.lora.lora_B' in param_name:
            continue

        # Vérifier si ce param appartient à un LinearWithLoRA (c'est le .base_layer.weight)
        # Exemple : "blocks.0.attention.q_proj.base_layer.weight"
        # → le module LoRA parent est "blocks.0.attention.q_proj"
        merged = False
        for lora_mod_name, lora_mod in lora_modules.items():
            base_weight_key = f'{lora_mod_name}.base_layer.weight'
            base_bias_key   = f'{lora_mod_name}.base_layer.bias'

            if param_name == base_weight_key:
                # Fusionner : W_merged = W_base + (lora_A @ lora_B).T * scaling
                W_base  = lora_mod.base_layer.weight.data.float()   # (out, in)
                lora_A  = lora_mod.lora.lora_A.data.float()         # (in, r)
                lora_B  = lora_mod.lora.lora_B.data.float()         # (r, out)
                delta   = (lora_A @ lora_B).T * lora_mod.lora.scaling  # (out, in)
                W_merged = (W_base + delta).to(torch.bfloat16).cpu()
                # Clé de sortie = sans le préfixe "base_layer."
                out_key = param_name.replace('.base_layer.weight', '.weight')
                merged_state[out_key] = W_merged
                merged = True
                break

            elif param_name == base_bias_key:
                out_key = param_name.replace('.base_layer.bias', '.bias')
                merged_state[out_key] = param.data.cpu()
                merged = True
                break

        if not merged:
            # Param normal (embedding, norm, etc.) → copie directe
            # Enlever le préfixe "base_layer." s'il reste dans le nom
            out_key = param_name.replace('.base_layer.', '.')
            if out_key not in merged_state:
                merged_state[out_key] = param.data.cpu()

    torch.save({
        'model_state_dict': merged_state,
        'config': {
            **config,
            'max_seq_len': MAX_SEQ_LEN,
            'yarn_scale':  2.0,
            'sft_step':    step,
            'sft_loss':    loss_val,
        },
        'step':           step,
        'last_loss':      loss_val,
        'saved_at':       datetime.now().isoformat(),
        'special_tokens': SPECIAL_TOKENS,
    }, OUTPUT_MERGED)

    elapsed  = time.time() - t0
    lora_mb  = os.path.getsize(OUTPUT_CKPT)   / 1e6
    merge_mb = os.path.getsize(OUTPUT_MERGED) / 1e6

    print(f"\n   💾 Save | step={step} | loss={loss_val:.4f} | durée={elapsed:.1f}s")
    print(f"      LoRA adapters  : {OUTPUT_CKPT}   ({lora_mb:.0f} MB)")
    print(f"      Merged model   : {OUTPUT_MERGED} ({merge_mb:.0f} MB)")


# ============================================
# FUNCTION CALLING VALIDATOR
# ============================================
def parse_tool_calls(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            data = json.loads(m.strip())
            if 'name' in data and 'args' in data:
                calls.append(data)
        except json.JSONDecodeError:
            pass
    return calls


@torch.no_grad()
def validate_function_calling(model, tokenizer, n=9):
    """
    ✅ FIX v9: comptage correct — total = nb de prompts tentés,
    valid = nb de prompts ayant produit au moins un tool_call valide.
    """
    model.eval()
    prompts = [
        "<|system|>You are a helpful assistant with tools.\n<|user|>Search for Python documentation<|end|><|assistant|>",
        "<|system|>You are a weather assistant.\n<|user|>What's the weather in Paris?<|end|><|assistant|>",
        "<|system|>You are a database assistant.\n<|user|>Find users with age > 25<|end|><|assistant|>",
    ]
    valid = 0
    total = 0
    for prompt in (prompts * math.ceil(n / len(prompts)))[:n]:
        total += 1
        try:
            tokens    = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokens], device=device)
            output    = model.generate(input_ids, max_new_tokens=150, temperature=0.7, top_k=40)
            text      = tokenizer.decode(output[0])
            if parse_tool_calls(text):
                valid += 1
        except Exception:
            pass
    model.train()
    return valid, total


# ============================================
# PRÉ-VÉRIFICATION RESUME (avant DataLoader)
# ============================================
IS_RESUME        = False
RESUME_BATCH_IDX = -1

if os.path.exists(OUTPUT_CKPT):
    _pre = torch.load(OUTPUT_CKPT, map_location='cpu')
    RESUME_BATCH_IDX = _pre.get('batch_idx', -1)
    if RESUME_BATCH_IDX == -1:
        _step            = _pre.get('step', 0)
        RESUME_BATCH_IDX = max(0, _step * GRAD_ACCUM - 1)
        print(f"\n📂 Checkpoint LoRA détecté → batch_idx recalculé depuis step={_step} → batch {RESUME_BATCH_IDX}")
    else:
        print(f"\n📂 Checkpoint LoRA détecté → resume au batch {RESUME_BATCH_IDX}")
    IS_RESUME = RESUME_BATCH_IDX > 0
    del _pre

# ============================================
# LOAD DATASET
# ============================================
train_dataset = SFTDataset(SFT_DATA_DIR, MAX_SEQ_LEN)
_shuffle      = not IS_RESUME
train_loader  = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = _shuffle,
    num_workers = 2,
    pin_memory  = True,
    collate_fn  = collate_fn,
    drop_last   = False,
)
if IS_RESUME:
    print(f"   ⚡ Resume mode: shuffle désactivé, reprise au batch {RESUME_BATCH_IDX}")

total_steps = math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS
print(f"   Steps total: {total_steps:,}")

# ============================================
# OPTIMIZER + SCHEDULER
# ============================================
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr           = LEARNING_RATE,
    weight_decay = 0.01,
    betas        = (0.9, 0.95),
)
scheduler = CosineScheduler(optimizer, LEARNING_RATE, total_steps)

# ============================================
# RESUME si checkpoint LoRA existant
# ============================================
global_step  = 0
current_loss = 0.0

if os.path.exists(OUTPUT_CKPT):
    print(f"\n📂 Chargement checkpoint LoRA pour resume...")
    lora_ckpt = torch.load(OUTPUT_CKPT, map_location='cpu')

    loaded = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in lora_ckpt['lora_state_dict']:
            param.data = lora_ckpt['lora_state_dict'][name].to(device)
            loaded += 1

    optimizer.load_state_dict(lora_ckpt['optimizer_state'])
    scheduler.current_step = lora_ckpt.get('scheduler_step', 0)
    global_step            = lora_ckpt.get('step', 0)
    current_loss           = lora_ckpt.get('last_loss', 0.0)
    RESUME_BATCH_IDX       = lora_ckpt.get('batch_idx', -1)
    IS_RESUME              = RESUME_BATCH_IDX >= 0

    print(f"   ✅ {loaded} poids LoRA rechargés")
    print(f"   📊 step={global_step} | loss={current_loss:.4f} | batch_idx={RESUME_BATCH_IDX}")
else:
    print(f"\n🆕 Démarrage SFT fresh")

print(f"   LR actuel : {scheduler.get_lr():.2e}")

# ============================================
# TRAINING LOOP
# ============================================
print(f"\n{'='*70}")
print(f"🚀 SFT TRAINING  |  {len(train_dataset):,} samples  |  {EPOCHS} epoch(s)")
print(f"   Save automatique toutes les {SAVE_EVERY_MINUTES} minutes")
print(f"{'='*70}")

model.train()
last_save_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print(f"{'='*70}")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, (input_ids, labels) in enumerate(pbar):

        # ── Skip rapide des batches déjà vus lors du resume ───
        if IS_RESUME and batch_idx <= RESUME_BATCH_IDX:
            if batch_idx % 500 == 0:
                print(f"\r   ⏭️  Skip {batch_idx}/{RESUME_BATCH_IDX}...", end='', flush=True)
            if batch_idx == RESUME_BATCH_IDX:
                print(f"\n   ✅ Reprise effective au batch {batch_idx + 1}")
                IS_RESUME = False   # ✅ FIX v9: désactiver le mode resume après le dernier skip
            continue

        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        # ── Forward + loss ─────────────────────────────────────
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n   ⚠️  NaN/Inf détecté au step {global_step} — batch skippé")
            optimizer.zero_grad(set_to_none=True)
            continue

        # ── Backward ───────────────────────────────────────────
        loss.backward()

        # ── Optimizer step tous les GRAD_ACCUM batches ─────────
        if (batch_idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step  += 1
            current_loss  = loss.item() * GRAD_ACCUM

            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr':   f'{scheduler.get_last_lr()[0]:.2e}',
                'step': f'{global_step}/{total_steps}',
            })

        # ── Auto-save toutes les 1h05 ───────────────────────────
        if time.time() - last_save_time >= SAVE_INTERVAL_SEC:
            save_checkpoint(model, optimizer, scheduler, global_step, current_loss, batch_idx=batch_idx)
            last_save_time = time.time()

    # ── Validation function calling après chaque epoch ─────────
    print(f"\n   🔍 Validating function calling...")
    valid, total = validate_function_calling(model, tokenizer, n=9)
    rate = 100 * valid / max(total, 1)
    print(f"   ✅ Tool calls: {valid}/{total} prompts valides ({rate:.0f}%)")

# ============================================
# SAVE FINAL
# ============================================
save_checkpoint(model, optimizer, scheduler, global_step, current_loss, batch_idx=-1)

print(f"\n{'='*70}")
print(f"✅ SFT TERMINÉ")
print(f"{'='*70}")
print(f"   Steps:          {global_step:,}")
print(f"   Last loss:      {current_loss:.4f}")
print(f"   LoRA adapters:  {OUTPUT_CKPT}")
print(f"   Merged model:   {OUTPUT_MERGED}")
print(f"\n🎯 NEXT: python chat.py")
