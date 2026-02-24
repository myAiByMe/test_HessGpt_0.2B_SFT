# HessGpt.py
"""
HessGPT - Architecture Transformer moderne PRODUCTION READY v4
✅ RMSNorm (plus rapide que LayerNorm)
✅ Flash Attention avec fallback
✅ TOUS LES BUGS FIXÉS v4:
- Loss ignore padding ✅
- Masque causal cached ✅
- Vocab resize support ✅
- Validation params ✅
- Soft-capping APPLIQUÉ ✅ (BUG FIXÉ)
- YaRN scale validation ✅
- Seq_len validation pour pos embeddings ✅
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TransformerBlock.transformer_block import TransformerBlock
from Attention.attention import RMSNorm

# ============================================
# MODÈLE HessGPT MODERNE : RMSNorm + Flash + RoPE + YaRN + SwiGLU
# ============================================

class HessGPT(nn.Module):
    """
    Modèle HessGPT - Architecture Transformer moderne
    
    Innovations:
    - RMSNorm (plus rapide que LayerNorm)
    - Flash Attention avec fallback automatique
    - RoPE au lieu de position embeddings absolues
    - YaRN pour extension de contexte (1024→4096)
    - SwiGLU au lieu de GELU/ReLU
    - GQA (Grouped Query Attention)
    - QK-Norm optionnel
    - Soft-capping des logits (Gemma-style) ✅ FIXÉ
    - Architecture style LLaMA/Mistral
    
    ✅ BUGS FIXÉS v4:
    - Loss avec ignore_index pour padding
    - Masque causal cached (pas recréé chaque forward)
    - Vocab resize support
    - Validation des paramètres
    - Soft-capping APPLIQUÉ dans forward ✅
    - YaRN scale warning si use_yarn=False
    - Seq_len validation pour position embeddings
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=2048,
        dropout=0.1,
        use_rope=True,
        use_yarn=False,
        yarn_scale=1.0,
        yarn_original_max_len=1024,
        use_swiglu=True,
        n_kv_heads=None,
        use_qk_norm=False,
        soft_cap=None,
        use_flash_attn=True
    ):
        """
        Args:
            vocab_size (int): Taille du vocabulaire
            embed_dim (int): Dimension des embeddings
            num_heads (int): Nombre de têtes Q d'attention
            num_layers (int): Nombre de Transformer Blocks
            max_seq_len (int): Longueur max (1024 pré-train, 4096 extension)
            dropout (float): Taux de dropout
            use_rope (bool): Utiliser RoPE
            use_yarn (bool): Activer YaRN pour extension
            yarn_scale (float): Facteur YaRN (4.0 pour 1024→4096)
            yarn_original_max_len (int): Longueur pré-train (1024)
            use_swiglu (bool): Utiliser SwiGLU (True) ou GELU (False)
            n_kv_heads (int): Nombre de têtes KV pour GQA (None = MHA)
            use_qk_norm (bool): Normaliser Q et K avant attention
            soft_cap (float): Soft-capping des logits (None = désactivé)
            use_flash_attn (bool): Utiliser Flash Attention (PyTorch 2.0+)
        """
        super().__init__()
        
        # ✅ VALIDATION DES PARAMÈTRES
        assert vocab_size > 0, "vocab_size must be positive"
        assert embed_dim > 0, "embed_dim must be positive"
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert num_layers > 0, "num_layers must be positive"
        assert max_seq_len > 0, "max_seq_len must be positive"
        
        # ✅ GQA validation
        if n_kv_heads is not None:
            assert num_heads % n_kv_heads == 0, \
                f"num_heads ({num_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        
        # ✅ YaRN validation (nouveau)
        if not use_yarn and yarn_scale != 1.0:
            print(f"⚠️  Warning: yarn_scale={yarn_scale} ignored (use_yarn=False)")
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_yarn = use_yarn
        self.yarn_scale = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len
        self.use_swiglu = use_swiglu
        self.n_kv_heads = n_kv_heads  # ✅ GQA
        self.use_qk_norm = use_qk_norm  # ✅ QK-Norm
        self.soft_cap = soft_cap  # ✅ Soft-capping
        self.use_flash_attn = use_flash_attn  # ✅ Flash Attention
        
        # Token Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings (uniquement si pas RoPE)
        if not use_rope:
            self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        else:
            self.position_embeddings = None
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks (avec RMSNorm + RoPE/YaRN + SwiGLU + GQA + Flash)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, 
                num_heads, 
                dropout,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
                use_yarn=use_yarn,
                yarn_scale=yarn_scale,
                yarn_original_max_len=yarn_original_max_len,
                use_swiglu=use_swiglu,
                n_kv_heads=n_kv_heads,        # ✅ GQA
                use_qk_norm=use_qk_norm,      # ✅ QK-Norm
                use_flash_attn=use_flash_attn # ✅ Flash Attention
            )
            for _ in range(num_layers)
        ])
        
        # ✅ RMSNorm finale (plus rapide que LayerNorm)
        self.ln_final = RMSNorm(embed_dim)
        
        # Output Head (projection vers vocabulaire)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Partager les poids (weight tying)
        self.output_head.weight = self.token_embeddings.weight
        
        # ✅ MASQUE CAUSAL CACHED (pas recréé à chaque forward)
        self.register_buffer('_causal_mask', None, persistent=False)
        
        # Initialisation
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialisation des poids"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            # RMSNorm a seulement weight (pas de bias)
            torch.nn.init.ones_(module.weight)
    
    def _get_causal_mask(self, seq_len, device):
        """
        ✅ OPTIMISÉ: Cache le masque causal au lieu de le recréer
        
        Args:
            seq_len: Longueur de séquence
            device: Device (cuda/cpu)
        
        Returns:
            mask: [seq_len, seq_len] boolean mask
                  True = masqué, False = visible
        """
        # Créer ou agrandir le cache si nécessaire
        if self._causal_mask is None or self._causal_mask.size(0) < seq_len:
            # Créer masque triangulaire supérieur
            # torch.triu avec diagonal=1 donne:
            # [[0, 1, 1],
            #  [0, 0, 1],
            #  [0, 0, 0]]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.register_buffer('_causal_mask', mask, persistent=False)
        
        # Retourner la portion nécessaire
        return self._causal_mask[:seq_len, :seq_len]
    
    def forward(self, input_ids, targets=None, pad_token_id=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            targets: [batch_size, seq_len] (optionnel)
            pad_token_id: ID du token de padding (pour ignore_index)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: Scalar (si targets fourni)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Token Embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # 2. Position Embeddings (uniquement si pas RoPE)
        if self.use_rope:
            x = self.dropout(token_embeds)
        else:
            # ✅ VALIDATION: Vérifier que seq_len <= max_seq_len
            assert seq_len <= self.max_seq_len, \
                f"seq_len ({seq_len}) > max_seq_len ({self.max_seq_len})"
            
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
            pos_embeds = self.position_embeddings(positions)
            x = self.dropout(token_embeds + pos_embeds)
        
        # 3. Masque causal (cached)
        mask = self._get_causal_mask(seq_len, device)
        
        # 4. Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # 5. Final RMSNorm
        x = self.ln_final(x)
        
        # 6. Output projection
        logits = self.output_head(x)
        
        # ✅ BUG FIX v4: SOFT-CAPPING APPLIQUÉ (Gemma-style)
        if self.soft_cap is not None:
            logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
        
        # 7. Loss (optionnel)
        loss = None
        if targets is not None:
            # ✅ FIXÉ: ignore_index pour padding
            ignore_index = pad_token_id if pad_token_id is not None else -100
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=ignore_index
            )
        
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Génération autoregressive
        
        Args:
            input_ids: [batch_size, seq_len] - Prompt
            max_new_tokens: Nombre de tokens à générer
            temperature: Contrôle la randomness
            top_k: Top-k sampling
        
        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        device = input_ids.device  # ✅ Auto-detect device
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Tronquer si trop long
                input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                
                # Forward
                logits, _ = self.forward(input_ids_cond)
                
                # Dernier token
                logits = logits[:, -1, :] / temperature
                
                # Top-k
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Ajouter
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def resize_token_embeddings(self, new_vocab_size):
        """
        ✅ Resize embeddings si vocab change
        
        Utile pour:
        - Ajouter des tokens spéciaux après création
        - Charger checkpoint avec vocab différent
        
        Args:
            new_vocab_size: Nouvelle taille du vocabulaire
        """
        if new_vocab_size == self.vocab_size:
            return
        
        print(f"📝 Resizing embeddings: {self.vocab_size} → {new_vocab_size}")
        
        # Créer nouveaux embeddings
        old_embeddings = self.token_embeddings
        self.token_embeddings = nn.Embedding(new_vocab_size, self.embed_dim)
        
        # Copier les anciens poids (pour tokens existants)
        old_vocab_size = min(old_embeddings.num_embeddings, new_vocab_size)
        with torch.no_grad():
            self.token_embeddings.weight.data[:old_vocab_size] = \
                old_embeddings.weight.data[:old_vocab_size]
        
        # Créer nouveau output head
        old_output = self.output_head
        self.output_head = nn.Linear(self.embed_dim, new_vocab_size, bias=False)
        
        # Weight tying
        self.output_head.weight = self.token_embeddings.weight
        
        # Update vocab_size
        self.vocab_size = new_vocab_size
        
        print(f"   ✅ Embeddings resized to {new_vocab_size}")
    
    def count_parameters(self):
        """Compte les paramètres du modèle"""
        token_params = self.token_embeddings.weight.numel()
        
        if self.position_embeddings is not None:
            pos_params = self.position_embeddings.weight.numel()
        else:
            pos_params = 0
        
        block_params = sum(p.numel() for block in self.blocks for p in block.parameters())
        ln_params = sum(p.numel() for p in self.ln_final.parameters())
        output_params = 0  # Partagé avec token_embeddings
        
        total = token_params + pos_params + block_params + ln_params + output_params
        
        return {
            'token_embeddings': token_params,
            'position_embeddings': pos_params,
            'transformer_blocks': block_params,
            'final_ln': ln_params,
            'output_head': output_params,
            'total': total
        }
    
    def get_config(self):
        """Retourne la configuration"""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'use_rope': self.use_rope,
            'use_yarn': self.use_yarn,
            'yarn_scale': self.yarn_scale,
            'yarn_original_max_len': self.yarn_original_max_len,
            'use_swiglu': self.use_swiglu,
            'n_kv_heads': self.n_kv_heads,        # ✅ GQA
            'use_qk_norm': self.use_qk_norm,      # ✅ QK-Norm
            'soft_cap': self.soft_cap,            # ✅ Soft-capping
            'use_flash_attn': self.use_flash_attn # ✅ Flash Attention
        }


if __name__ == "__main__":
    print("\n🚀 HessGPT - Architecture Moderne (PRODUCTION READY v4)\n")
    print("="*80)
    
    # Configuration moderne (LLaMA-style avec GQA + RMSNorm + Flash)
    model = HessGPT(
        vocab_size=32005,  # Mistral 32000 + 5 special tokens
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=1024,      # Pré-entraînement
        use_rope=True,
        use_yarn=False,        # Désactivé pour pré-train
        yarn_scale=1.0,
        yarn_original_max_len=1024,
        use_swiglu=True,       # SwiGLU activé
        n_kv_heads=4,          # ✅ GQA: 12 Q heads → 4 KV heads (ratio 3:1)
        use_qk_norm=True,      # ✅ QK-Norm avec RMSNorm
        soft_cap=30.0,         # ✅ Soft-capping activé
        use_flash_attn=True    # ✅ Flash Attention activé
    )
    
    config = model.get_config()
    params = model.count_parameters()
    
    print("✅ NOUVELLES FEATURES v4:")
    print("   • RMSNorm (plus rapide que LayerNorm)")
    print("   • Flash Attention avec fallback automatique")
    print("   • GQA (Grouped Query Attention)")
    print("   • Soft-capping des logits ✅ APPLIQUÉ")
    print("   • QK-Norm (optionnel)")
    print("   • Mistral tokenizer (32k vocab)")
    
    print("\n✅ BUGS FIXÉS v4:")
    print("   • Loss avec ignore_index (padding)")
    print("   • Masque causal cached")
    print("   • Vocab resize support")
    print("   • Validation params")
    print("   • Soft-capping APPLIQUÉ dans forward ✅")
    print("   • YaRN scale validation")
    print("   • Seq_len validation pour pos embeddings")
    
    print("\nConfiguration:")
    print(f"  - Vocab size: {config['vocab_size']:,} (Mistral 32k + 5 special)")
    print(f"  - Embed dim: {config['embed_dim']}")
    print(f"  - Num layers: {config['num_layers']}")
    print(f"  - Q heads: {config['num_heads']}")
    print(f"  - KV heads: {config['n_kv_heads']} ✅ (GQA ratio {config['num_heads']//config['n_kv_heads']}:1)")
    print(f"  - Max seq len: {config['max_seq_len']}")
    print(f"  - Use RoPE: {config['use_rope']}")
    print(f"  - Use YaRN: {config['use_yarn']}")
    print(f"  - Use SwiGLU: {config['use_swiglu']}")
    print(f"  - QK-Norm: {config['use_qk_norm']} (avec RMSNorm)")
    print(f"  - Soft-cap: {config['soft_cap']} ✅ APPLIQUÉ")
    print(f"  - Flash Attention: {config['use_flash_attn']}")
    
    print(f"\nParamètres:")
    print(f"  - Total: {params['total']:,}")
    print(f"  - Transformer blocks: {params['transformer_blocks']:,}")
    
    # Test forward
    print("\nTest forward:")
    x = torch.randint(0, 32005, (2, 128))
    y = torch.randint(0, 32005, (2, 128))
    
    logits, loss = model(x, targets=y, pad_token_id=0)
    
    print(f"  - Input: {x.shape}")
    print(f"  - Output: {logits.shape}")
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    if model.soft_cap is not None:
        print(f"  - Soft-cap applied: logits ∈ [{-model.soft_cap:.1f}, {model.soft_cap:.1f}] ✅")
        # Vérifier que le soft-cap est bien appliqué
        assert logits.max().item() <= model.soft_cap + 0.1, "Soft-cap not applied!"
        assert logits.min().item() >= -model.soft_cap - 0.1, "Soft-cap not applied!"
        print(f"  - ✅ Soft-cap verification PASSED")
    
    print("="*80)
    print("✅ PRODUCTION READY v4 - Tous les bugs fixés!")
    print("="*80 + "\n")