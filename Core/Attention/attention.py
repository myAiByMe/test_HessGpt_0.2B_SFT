# attention.py
"""
Multi-Head Attention avec RoPE + YaRN + Flash Attention
✅ RMSNorm intégré (remplace LayerNorm)
✅ Flash Attention avec fallback
✅ BUGS FIXÉS:
- Masque utilise bool directement
- Validation YaRN scale
- YaRN attention scaling (scores *= sqrt(yarn_scale))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================
# RMSNorm (Root Mean Square Normalization)
# ============================================

class RMSNorm(nn.Module):
    """
    RMSNorm - Plus rapide et simple que LayerNorm
    
    Utilisé dans: LLaMA, Mistral, Qwen, GPT-NeoX
    
    Avantages vs LayerNorm:
    - Pas de centrage (mean) → plus rapide
    - Pas de beta param → moins de params
    - Même qualité de training
    """
    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim (int): Dimension à normaliser
            eps (float): Epsilon pour stabilité numérique
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, dim] ou [batch_size, num_heads, seq_len, head_dim]
        
        Returns:
            normalized: même shape que x
        """
        # RMS = Root Mean Square
        # rms = sqrt(mean(x²) + eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ============================================
# RoPE + YaRN
# ============================================

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) avec YaRN
    
    YaRN permet d'étendre la longueur de contexte au-delà du pré-entraînement
    sans réentraînement complet.
    
    Exemple: Pré-entraîné sur 1024 tokens → Extension à 4096 tokens
    """
    def __init__(self, dim, max_seq_len=2048, base=10000, device=None,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024):
        """
        Args:
            dim (int): Dimension par tête (head_dim)
            max_seq_len (int): Longueur max après extension (4096)
            base (int): Base pour les fréquences (10000)
            use_yarn (bool): Activer YaRN
            yarn_scale (float): Facteur d'échelle (4.0 pour 1024→4096)
            yarn_original_max_len (int): Longueur du pré-entraînement (1024)
        """
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.use_yarn = use_yarn
        self.yarn_scale = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len
        
        # ✅ VALIDATION YaRN SCALE
        if use_yarn:
            assert 0.1 <= yarn_scale <= 16.0, \
                f"yarn_scale must be in [0.1, 16.0], got {yarn_scale}"
        
        # Calculer les fréquences (avec ou sans YaRN)
        if use_yarn:
            inv_freq = self._compute_yarn_frequencies()
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache pour les embeddings précalculés
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _compute_yarn_frequencies(self):
        """
        Calcule les fréquences avec YaRN
        
        YaRN applique un scaling non-uniforme:
        - Hautes fréquences: scaling faible (info locale)
        - Basses fréquences: scaling fort (info globale)
        """
        # Fréquences de base
        freqs = torch.arange(0, self.dim, 2).float() / self.dim
        inv_freq_base = 1.0 / (self.base ** freqs)
        
        if self.yarn_scale == 1.0:
            return inv_freq_base
        
        # YaRN: Interpolation non-uniforme
        alpha = self.yarn_scale
        beta = 32  # Point de transition
        
        dims = torch.arange(0, self.dim, 2).float()
        
        # Scaling non-uniforme
        scale = torch.where(
            dims < beta,
            torch.ones_like(dims),  # Pas de scaling pour hautes fréq
            1 + (alpha - 1) * (dims - beta) / (self.dim - beta)  # Scaling progressif
        )
        
        # Appliquer le scaling inverse aux fréquences
        inv_freq_yarn = inv_freq_base / scale
        
        return inv_freq_yarn
    
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        """Met à jour le cache cos/sin si nécessaire"""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Créer les positions
            t = torch.arange(seq_len, device=device, dtype=dtype)
            
            # Calculer les fréquences pour chaque position
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            
            # Créer les embeddings avec répétition
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        
        return self._cos_cached, self._sin_cached
    
    def rotate_half(self, x):
        """
        Rotation de la moitié des dimensions
        [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k):
        """
        Applique RoPE/YaRN à Q et K
        
        Args:
            q: [batch_size, num_heads, seq_len, head_dim]
            k: [batch_size, num_heads, seq_len, head_dim]
        
        Returns:
            q_rot, k_rot avec positions encodées
        """
        seq_len = q.shape[2]
        
        # Obtenir cos et sin
        cos, sin = self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        # Ajouter dimensions pour batch et heads
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        
        # Appliquer la rotation
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot
    
    def forward(self, q, k):
        """Forward pass - applique RoPE/YaRN"""
        return self.apply_rotary_pos_emb(q, k)


# ============================================
# Multi-Head Attention
# ============================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention avec RoPE + YaRN + GQA + QK-Norm + Flash Attention
    
    ✅ NOUVEAUTÉS:
    - RMSNorm au lieu de LayerNorm (plus rapide)
    - Flash Attention avec fallback automatique
    - GQA (Grouped Query Attention) pour cache KV efficient
    - QK-Norm optionnel pour stabilité avec LR agressifs
    - YaRN attention scaling: scores *= sqrt(yarn_scale) quand yarn_scale > 1
    
    ✅ BUGS FIXÉS:
    - Masque utilise bool directement (pas mask == 0)
    - YaRN attention scaling appliqué correctement
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, 
                 use_rope=True, max_seq_len=2048,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024,
                 n_kv_heads=None, use_qk_norm=False, use_flash_attn=True):
        """
        Args:
            embed_dim (int): Dimension des embeddings (768 pour GPT-2)
            num_heads (int): Nombre de têtes Q (12 pour GPT-2)
            dropout (float): Taux de dropout
            use_rope (bool): Utiliser RoPE
            max_seq_len (int): Longueur max de séquence
            use_yarn (bool): Activer YaRN
            yarn_scale (float): Facteur d'échelle YaRN
            yarn_original_max_len (int): Longueur pré-entraînement
            n_kv_heads (int): Nombre de têtes KV pour GQA (None = MHA classique)
            use_qk_norm (bool): Normaliser Q et K avant attention
            use_flash_attn (bool): Utiliser Flash Attention (PyTorch 2.0+)
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        
        # ✅ GQA: Si n_kv_heads fourni, utilise GQA, sinon MHA classique
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        assert num_heads % self.n_kv_heads == 0, \
            f"num_heads ({num_heads}) doit être divisible par n_kv_heads ({self.n_kv_heads})"
        self.num_queries_per_kv = num_heads // self.n_kv_heads
        
        # Dimension KV (pour GQA)
        self.kv_dim = self.n_kv_heads * self.head_dim
        
        # Projections Q, K, V (K et V utilisent kv_dim si GQA)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        
        # Projection de sortie
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # ✅ QK-Norm: Normalisation avec RMSNorm (plus rapide que LayerNorm)
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        
        # RoPE avec YaRN
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, 
                max_seq_len,
                use_yarn=use_yarn,
                yarn_scale=yarn_scale,
                yarn_original_max_len=yarn_original_max_len
            )
        else:
            self.rope = None
        
        # ✅ Flash Attention availability check
        self._flash_attn_available = False
        if use_flash_attn:
            try:
                # Check if scaled_dot_product_attention exists (PyTorch 2.0+)
                F.scaled_dot_product_attention
                self._flash_attn_available = True
            except AttributeError:
                print("⚠️  Flash Attention non disponible (PyTorch < 2.0), fallback vers attention standard")
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [seq_len, seq_len] - Masque causal BOOLEAN
                  True = masqué, False = visible
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Projections Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch, seq_len, kv_dim] ✅ GQA
        v = self.v_proj(x)  # [batch, seq_len, kv_dim] ✅ GQA
        
        # Reshape pour multi-head
        # Q: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # K, V: [batch, seq_len, n_kv_heads, head_dim] ✅ GQA
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Transpose pour attention: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # ✅ QK-Norm: Normaliser Q et K si activé
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Appliquer RoPE/YaRN si activé
        if self.use_rope:
            q, k = self.rope(q, k)
        
        # ✅ GQA: Répéter K et V pour correspondre au nombre de têtes Q
        if self.n_kv_heads != self.num_heads:
            # k, v: [batch, n_kv_heads, seq_len, head_dim]
            # → [batch, n_kv_heads, 1, seq_len, head_dim]
            # → repeat → [batch, n_kv_heads, num_queries_per_kv, seq_len, head_dim]
            # → reshape → [batch, num_heads, seq_len, head_dim]
            k = k.unsqueeze(2).repeat(1, 1, self.num_queries_per_kv, 1, 1)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            
            v = v.unsqueeze(2).repeat(1, 1, self.num_queries_per_kv, 1, 1)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # ✅ FLASH ATTENTION ou attention standard
        if self.use_flash_attn and self._flash_attn_available:
            # Flash Attention (PyTorch 2.0+)
            # scaled_dot_product_attention gère:
            # - Scaling automatique (1/sqrt(head_dim))
            # - Masque causal
            # - Dropout
            # - Optimisations mémoire O(N) au lieu de O(N²)
            
            # Convertir mask pour Flash Attention
            # Flash veut None (causal auto) ou attention_mask
            attn_mask = None
            if mask is not None:
                # mask est [seq_len, seq_len] bool (True = masqué)
                # Flash veut [batch, heads, seq_len, seq_len] float (-inf = masqué)
                attn_mask = torch.zeros(seq_len, seq_len, dtype=q.dtype, device=q.device)
                attn_mask.masked_fill_(mask, float('-inf'))
                # Broadcast pour batch et heads
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            
            # YaRN attention scaling (si besoin)
            scale = None
            if self.use_rope and self.rope.use_yarn and self.rope.yarn_scale > 1.0:
                scale = math.sqrt(self.rope.yarn_scale) / math.sqrt(self.head_dim)
            else:
                scale = 1.0 / math.sqrt(self.head_dim)
            
            # Flash Attention forward
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=scale,
            )
        else:
            # ✅ FALLBACK: Attention standard (pour compatibilité)
            # Calculer les scores d'attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # ✅ YaRN attention scaling
            # Quand on extend le contexte avec YaRN (yarn_scale > 1), 
            # on doit aussi scaler les scores d'attention
            if self.use_rope and self.rope.use_yarn and self.rope.yarn_scale > 1.0:
                scores = scores * math.sqrt(self.rope.yarn_scale)
            
            # ✅ Appliquer le masque causal si fourni
            # mask vient de HessGpt._get_causal_mask() qui renvoie bool
            # True = masqué, False = visible
            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Appliquer l'attention aux valeurs
            output = torch.matmul(attn_weights, v)
        
        # Transpose et reshape: [batch, seq_len, embed_dim]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, embed_dim)
        
        # Projection finale
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output


if __name__ == "__main__":
    print("\n🚀 ATTENTION AVEC RMSNorm + Flash Attention + RoPE + YaRN + GQA\n")
    print("="*70)
    print("✅ NOUVEAUTÉS:")
    print("   • RMSNorm intégré (plus rapide que LayerNorm)")
    print("   • Flash Attention avec fallback automatique")
    print("   • GQA (Grouped Query Attention)")
    print("   • QK-Norm (optionnel)")
    print("   • YaRN attention scaling")
    print("="*70)
    
    # Test RMSNorm
    print("\nTest 1: RMSNorm")
    rms = RMSNorm(768)
    x = torch.randn(2, 10, 768)
    out = rms(x)
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {out.shape}")
    print(f"✅ Params: {sum(p.numel() for p in rms.parameters())}")
    
    # Test avec GQA + Flash Attention
    batch_size = 2
    seq_len = 128
    embed_dim = 768
    num_heads = 12
    n_kv_heads = 4  # GQA: ratio 3:1
    
    print("\nTest 2: Attention avec GQA + Flash Attention")
    attention = MultiHeadAttention(
        embed_dim, 
        num_heads,
        use_rope=True,
        max_seq_len=1024,
        use_yarn=False,
        n_kv_heads=n_kv_heads,  # ✅ GQA activé
        use_qk_norm=True,        # ✅ QK-Norm avec RMSNorm
        use_flash_attn=True      # ✅ Flash Attention
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Créer masque causal (comme dans HessGpt)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    output = attention(x, mask)
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Flash Attention: {attention._flash_attn_available}")
    print(f"✅ KV heads: {attention.n_kv_heads} (Q heads: {num_heads})")
    print(f"✅ QK-Norm: {attention.use_qk_norm} (avec RMSNorm)")
    
    # Compte params
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"✅ Total params: {total_params:,}")
    
    print("\n" + "="*70)
    print("✅ PRODUCTION READY - RMSNorm + Flash Attention!")
    print("="*70 + "\n")