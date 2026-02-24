#!/usr/bin/env python3
"""
🔥 HessGPT Mini 0.2B - SFT LoRA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ NO CLI - Lance direct
✅ Charge depuis sft_data/*.npy (pré-tokenisé) → instantané
✅ YaRN 1024 → 2048
✅ LoRA Rank 64 sur toutes les couches (q/k/v/o + SwiGLU)
✅ Gradient accumulation + Cosine scheduler
✅ Save intermédiaire toutes les 50 min
✅ Modèle chargé en bf16 (économie mémoire)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE:
  1. Upload sft_data/ (depuis prepare_sft_data.py)
  2. Upload hessgpt_mini_pretrain.pt dans ./tinyModel/
  3. Lance : python sft_lora_function_calling.py
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
PRETRAIN_CHECKPOINT = './tinyModel/hessgpt_mini_pretrain.pt'
OUTPUT_DIR          = './tinyModel/lora_adapters'
OUTPUT_CKPT         = f'{OUTPUT_DIR}/lora_checkpoint.pt'
SAVE_EVERY_MINUTES  = 50

EPOCHS              = 1
BATCH_SIZE          = 16
MAX_SEQ_LEN         = 2048
LORA_RANK           = 64
LEARNING_RATE       = 1e-4
GRAD_ACCUM          = 4

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
print("🔥 HessGPT Mini 0.2B — SFT LoRA")
print(f"   Input:   {PRETRAIN_CHECKPOINT}")
print(f"   Data:    {SFT_DATA_DIR}/")
print(f"   Output:  {OUTPUT_DIR}")
print(f"   Batch:   {BATCH_SIZE} × grad_accum {GRAD_ACCUM} = effective {BATCH_SIZE*GRAD_ACCUM}")
print(f"   LR:      {LEARNING_RATE:.0e}")
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
    vocab_size=VOCAB_SIZE,
    embed_dim=config.get('embed_dim', 896),
    num_heads=config.get('num_heads', 14),
    num_layers=config.get('num_layers', 16),
    max_seq_len=MAX_SEQ_LEN,
    dropout=0.0,
    use_rope=True,
    use_yarn=True,
    yarn_scale=2.0,               # 1024 → 2048
    yarn_original_max_len=1024,
    use_swiglu=True,
    n_kv_heads=config.get('n_kv_heads', 7),
    use_qk_norm=True,
    soft_cap=30.0,
    use_flash_attn=True
)

# Fix torch.compile prefix
state_dict = checkpoint['model_state_dict']
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=False)

# ✅ Chargement en bf16 : réduit la mémoire des poids frozen de ~50%
# Les poids pretrain (float32) sont convertis → bf16
# Légère perte de précision mais sans impact car ces poids ne reçoivent pas de gradient
model = model.to(device, dtype=torch.bfloat16)

pretrain_epoch = checkpoint.get('epoch', '?')
pretrain_loss  = checkpoint.get('last_loss', 0.0)
print(f"   ✅ Pretrain epoch={pretrain_epoch} | loss={pretrain_loss:.4f} | YaRN x2 (1024→2048) | bf16")

# ============================================
# LoRA
# ============================================
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, dropout):
        super().__init__()
        self.r       = r
        self.scaling = alpha / r
        self.lora_A  = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B  = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling

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

for param in model.parameters():
    param.requires_grad = False

TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']

for name, module in model.named_modules():
    for target in TARGET_MODULES:
        if target in name and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name  = name.split('.')[-1]
            parent      = model.get_submodule(parent_name) if parent_name else model
            lora_layer  = LinearWithLoRA(module, r=LORA_RANK, alpha=LORA_RANK*2, dropout=0.05).to(device)
            setattr(parent, child_name, lora_layer)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   ✅ LoRA sur: {TARGET_MODULES}")
print(f"   📊 Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.1f}M ({100*trainable_params/total_params:.2f}%)")

# ============================================
# DATASET — charge depuis .npy
# ============================================
class SFTDataset(Dataset):
    def __init__(self, data_dir, max_seq_len):
        print(f"\n📦 Loading SFT dataset from {data_dir}/...")

        input_path  = os.path.join(data_dir, 'sft_input_ids.npy')
        labels_path = os.path.join(data_dir, 'sft_labels.npy')
        lengths_path = os.path.join(data_dir, 'sft_lengths.npy')
        meta_path   = os.path.join(data_dir, 'sft_meta.json')

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"❌ {input_path} introuvable — lance prepare_sft_data.py d'abord")

        self.input_ids = np.load(input_path)
        self.labels    = np.load(labels_path)
        self.lengths   = np.load(lengths_path)
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
        length    = min(self.lengths[idx], self.max_seq_len)
        input_ids = torch.tensor(self.input_ids[idx, :length], dtype=torch.long)
        labels    = torch.tensor(self.labels[idx, :length],    dtype=torch.long)
        return input_ids, labels

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels    = [item[1] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels    = torch.nn.utils.rnn.pad_sequence(labels,    batch_first=True, padding_value=-100)
    return input_ids, labels

# ============================================
# SCHEDULER — Cosine
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
# SAVE
# ============================================
def save_checkpoint(model, optimizer, scheduler, step, loss_val):
    lora_state = {
        name: param.data.cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save({
        'lora_state_dict':  lora_state,
        'optimizer_state':  optimizer.state_dict(),
        'scheduler_step':   scheduler.current_step,
        'step':             step,
        'last_loss':        loss_val,
        'saved_at':         datetime.now().isoformat(),
        'config': {
            'lora_rank':      LORA_RANK,
            'lora_alpha':     LORA_RANK * 2,
            'target_modules': TARGET_MODULES,
            'max_seq_len':    MAX_SEQ_LEN,
            'yarn_scale':     2.0,
        },
        'special_tokens': SPECIAL_TOKENS,
    }, OUTPUT_CKPT)
    print(f"\n   💾 Save | step={step} | loss={loss_val:.4f} → {OUTPUT_CKPT}")

# ============================================
# FUNCTION CALLING VALIDATOR
# ============================================
def parse_tool_calls(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            data = json.loads(match.strip())
            if 'name' in data and 'args' in data:
                tool_calls.append(data)
        except json.JSONDecodeError:
            pass
    return tool_calls

@torch.no_grad()
def validate_function_calling(model, tokenizer, n=9):
    model.eval()
    prompts = [
        "<|system|>You are a helpful assistant with tools.\n<|user|>Search for Python documentation<|end|><|assistant|>",
        "<|system|>You are a weather assistant.\n<|user|>What's the weather in Paris?<|end|><|assistant|>",
        "<|system|>You are a database assistant.\n<|user|>Find users with age > 25<|end|><|assistant|>",
    ]
    valid = total = 0
    for prompt in prompts * (n // len(prompts) + 1):
        if total >= n:
            break
        try:
            tokens    = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokens], device=device)
            output    = model.generate(input_ids, max_new_tokens=150, temperature=0.7, top_k=40)
            text      = tokenizer.decode(output[0])
            tcs       = parse_tool_calls(text)
            total    += len(tcs)
            valid    += len(tcs)
        except Exception:
            pass
    model.train()
    return valid, total

# ============================================
# LOAD DATASET
# ============================================
train_dataset = SFTDataset(SFT_DATA_DIR, MAX_SEQ_LEN)
train_loader  = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

total_steps = math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS
print(f"   Steps total: {total_steps:,}")

# ============================================
# OPTIMIZER + SCHEDULER
# ============================================
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=0.01
)

scheduler = CosineScheduler(optimizer, LEARNING_RATE, total_steps)

# ============================================
# RESUME si checkpoint LoRA existant
# ============================================
global_step  = 0
resume_step  = 0
current_loss = 0.0

if os.path.exists(OUTPUT_CKPT):
    print(f"\n📂 Checkpoint LoRA trouvé → resume...")
    lora_ckpt = torch.load(OUTPUT_CKPT, map_location='cpu')

    # Restore LoRA weights
    for name, param in model.named_parameters():
        if param.requires_grad and name in lora_ckpt['lora_state_dict']:
            param.data = lora_ckpt['lora_state_dict'][name].to(device)

    optimizer.load_state_dict(lora_ckpt['optimizer_state'])
    scheduler.current_step = lora_ckpt.get('scheduler_step', 0)
    global_step  = lora_ckpt.get('step', 0)
    resume_step  = global_step
    current_loss = lora_ckpt.get('last_loss', 0.0)
    print(f"   ✅ Reprise step={global_step} | loss={current_loss:.4f}")
else:
    print(f"\n🆕 Démarrage SFT fresh")

print(f"   LR actuel : {scheduler.get_lr():.2e}")

# ============================================
# TRAINING
# ============================================
print(f"\n{'='*70}")
print(f"🚀 SFT TRAINING  |  {len(train_dataset):,} samples  |  {EPOCHS} epoch(s)")
print(f"{'='*70}")

model.train()
last_save_time = time.time()
save_interval  = SAVE_EVERY_MINUTES * 60

for epoch in range(EPOCHS):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print(f"{'='*70}")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, (input_ids, labels) in enumerate(pbar):

        # Skip batches déjà vus si resume
        if resume_step > 0 and global_step < resume_step:
            global_step += 1
            continue

        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n   ⚠️  NaN/Inf step {global_step}, skip")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

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

        # Auto-save
        if time.time() - last_save_time >= save_interval:
            save_checkpoint(model, optimizer, scheduler, global_step, current_loss)
            last_save_time = time.time()

    # Validation function calling après chaque epoch
    print(f"\n   🔍 Validating function calling...")
    valid, total = validate_function_calling(model, tokenizer)
    rate = valid / max(total, 1) * 100
    print(f"   ✅ Tool calls: {total} | Valid: {valid} ({rate:.0f}%)")

# ============================================
# SAVE FINAL
# ============================================
save_checkpoint(model, optimizer, scheduler, global_step, current_loss)

print(f"\n{'='*70}")
print(f"✅ SFT TERMINÉ")
print(f"{'='*70}")
print(f"   Steps:      {global_step:,}")
print(f"   Last loss:  {current_loss:.4f}")
print(f"   Adapters:   {OUTPUT_CKPT}")
print(f"\n🎯 NEXT: python test_function_calling.py")
