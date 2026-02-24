# transformer_block.py
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Attention.attention import MultiHeadAttention, RMSNorm
from FeedForward.feedforward import FeedForward

# ============================================
# TRANSFORMER BLOCK AVEC RMSNorm + RoPE + YaRN + SwiGLU
# ============================================

class TransformerBlock(nn.Module):
    """
    Un bloc Transformer complet moderne avec:
    - RMSNorm (plus rapide que LayerNorm)
    - RoPE + YaRN pour les positions
    - SwiGLU pour l'activation (au lieu de GELU)
    - GQA (Grouped Query Attention)
    - QK-Norm optionnel
    - Flash Attention avec fallback
    
    Architecture:
    1. RMSNorm → Multi-Head Attention (RoPE/YaRN/GQA/Flash) → Dropout → Residual
    2. RMSNorm → Feed-Forward (SwiGLU) → Dropout → Residual
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, 
                 use_rope=True, max_seq_len=2048,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024,
                 use_swiglu=True, n_kv_heads=None, use_qk_norm=False, use_flash_attn=True):
        """
        Args:
            embed_dim (int): Dimension des embeddings
            num_heads (int): Nombre de têtes d'attention
            dropout (float): Taux de dropout
            use_rope (bool): Utiliser RoPE
            max_seq_len (int): Longueur max de séquence
            use_yarn (bool): Activer YaRN pour extension de contexte
            yarn_scale (float): Facteur d'échelle YaRN (4.0 pour 1024→4096)
            yarn_original_max_len (int): Longueur pré-entraînement (1024)
            use_swiglu (bool): Utiliser SwiGLU (True) ou GELU (False)
            n_kv_heads (int): Nombre de têtes KV pour GQA (None = MHA classique)
            use_qk_norm (bool): Normaliser Q et K avant attention
            use_flash_attn (bool): Utiliser Flash Attention (PyTorch 2.0+)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        self.n_kv_heads = n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        
        # ✅ RMSNorm (remplace LayerNorm - plus rapide)
        self.ln1 = RMSNorm(embed_dim)
        
        # Multi-Head Attention (avec RoPE/YaRN/GQA/Flash)
        self.attention = MultiHeadAttention(
            embed_dim, 
            num_heads, 
            dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            use_yarn=use_yarn,
            yarn_scale=yarn_scale,
            yarn_original_max_len=yarn_original_max_len,
            n_kv_heads=n_kv_heads,        # ✅ GQA
            use_qk_norm=use_qk_norm,      # ✅ QK-Norm
            use_flash_attn=use_flash_attn # ✅ Flash Attention
        )
        
        # ✅ RMSNorm (remplace LayerNorm - plus rapide)
        self.ln2 = RMSNorm(embed_dim)
        
        # Feed-Forward Network (avec SwiGLU)
        self.ffn = FeedForward(embed_dim, dropout, use_swiglu=use_swiglu)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [seq_len, seq_len] - Masque causal
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # 1. Attention block avec residual connection
        # Pre-RMSNorm (GPT-2/GPT-3/LLaMA utilise pre-norm)
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = residual + x  # Residual connection
        
        # 2. Feed-Forward block avec residual connection
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x  # Residual connection
        
        return x


def create_causal_mask(seq_len, device='cpu'):
    """
    Crée un masque causal triangulaire
    
    Args:
        seq_len (int): Longueur de la séquence
        device (str/torch.device): Device
    
    Returns:
        mask: [seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


if __name__ == "__main__":
    print("\n🚀 TRANSFORMER BLOCK - RMSNorm + Flash Attention + RoPE + YaRN + SwiGLU + GQA\n")
    print("="*70)
    
    # Test du bloc complet
    batch_size = 2
    seq_len = 128
    embed_dim = 768
    num_heads = 12
    n_kv_heads = 4  # GQA
    
    print("Configuration moderne (LLaMA-style avec GQA + Flash Attention):")
    block = TransformerBlock(
        embed_dim,
        num_heads,
        use_rope=True,
        max_seq_len=1024,
        use_yarn=False,          # Pré-entraînement
        use_swiglu=True,         # SwiGLU activé
        n_kv_heads=n_kv_heads,   # ✅ GQA activé
        use_qk_norm=True,        # ✅ QK-Norm avec RMSNorm
        use_flash_attn=True      # ✅ Flash Attention
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = create_causal_mask(seq_len)
    
    output = block(x, mask)
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Paramètres: {sum(p.numel() for p in block.parameters()):,}")
    print(f"✅ RMSNorm: True (plus rapide que LayerNorm)")
    print(f"✅ SwiGLU: {block.use_swiglu}")
    print(f"✅ GQA: {block.n_kv_heads} KV heads ({num_heads} Q heads)")
    print(f"✅ QK-Norm: {block.use_qk_norm}")
    print(f"✅ Flash Attention: {block.use_flash_attn}")
    print("="*70 + "\n")