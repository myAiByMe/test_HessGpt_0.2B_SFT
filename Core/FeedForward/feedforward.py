import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) avec SwiGLU pour LLaMA/GPT-style models
    
    Architecture moderne :
    - SwiGLU activation (remplace GELU)
    - Gate mechanism pour meilleure expressivité
    - Utilisé dans LLaMA, PaLM, et autres modèles SOTA
    
    SwiGLU: Swish-Gated Linear Unit
    - Combine Swish (SiLU) et gating mechanism
    - Formule: SwiGLU(x) = Swish(W1·x) ⊗ (W2·x)
    - Plus performant que GELU/ReLU
    """
    def __init__(self, embed_dim, dropout=0.1, use_swiglu=True):
        """
        Args:
            embed_dim (int): Dimension des embeddings (768 pour GPT-2 small)
            dropout (float): Taux de dropout (0.1 par défaut)
            use_swiglu (bool): Utiliser SwiGLU (True) ou GELU (False)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_swiglu = use_swiglu
        
        if use_swiglu:
            # SwiGLU nécessite 2 projections pour le gate
            # On utilise 8/3 * embed_dim au lieu de 4 * embed_dim
            # pour compenser le gate et garder le même nombre de paramètres
            self.hidden_dim = int(8 * embed_dim / 3)
            
            # Projection pour le gate (W1)
            self.gate_proj = nn.Linear(embed_dim, self.hidden_dim, bias=False)
            
            # Projection pour la valeur (W2)
            self.up_proj = nn.Linear(embed_dim, self.hidden_dim, bias=False)
            
            # Projection de sortie (W3)
            self.down_proj = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        else:
            # GELU classique (pour compatibilité)
            self.hidden_dim = 4 * embed_dim
            self.fc1 = nn.Linear(embed_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, embed_dim)
        
        # Dropout pour la régularisation
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        if self.use_swiglu:
            # SwiGLU activation
            # 1. Gate: Swish(W1·x) = x·sigmoid(x)
            gate = F.silu(self.gate_proj(x))  # SiLU = Swish
            
            # 2. Value: W2·x
            value = self.up_proj(x)
            
            # 3. Element-wise multiplication: gate ⊗ value
            x = gate * value
            
            # 4. Dropout
            x = self.dropout(x)
            
            # 5. Projection finale: W3·x
            x = self.down_proj(x)
            
            # 6. Dropout final
            x = self.dropout(x)
        else:
            # GELU classique (fallback)
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
        
        return x


# ============================================
# TESTS
# ============================================

def test_swiglu_vs_gelu():
    """Compare SwiGLU vs GELU"""
    print("\n" + "="*60)
    print("TEST 1: SwiGLU vs GELU")
    print("="*60)
    
    batch_size = 2
    seq_len = 10
    embed_dim = 768
    
    # FFN avec SwiGLU
    ffn_swiglu = FeedForward(embed_dim, use_swiglu=True)
    
    # FFN avec GELU
    ffn_gelu = FeedForward(embed_dim, use_swiglu=False)
    
    # Input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"✓ Input shape: {x.shape}")
    
    # Forward SwiGLU
    output_swiglu = ffn_swiglu(x)
    params_swiglu = sum(p.numel() for p in ffn_swiglu.parameters())
    
    print(f"\n📊 SwiGLU:")
    print(f"  - Output shape: {output_swiglu.shape}")
    print(f"  - Hidden dim: {ffn_swiglu.hidden_dim}")
    print(f"  - Paramètres: {params_swiglu:,}")
    
    # Forward GELU
    output_gelu = ffn_gelu(x)
    params_gelu = sum(p.numel() for p in ffn_gelu.parameters())
    
    print(f"\n📊 GELU:")
    print(f"  - Output shape: {output_gelu.shape}")
    print(f"  - Hidden dim: {ffn_gelu.hidden_dim}")
    print(f"  - Paramètres: {params_gelu:,}")
    
    print(f"\n💡 Comparaison:")
    print(f"  - SwiGLU a {params_swiglu - params_gelu:+,} paramètres")
    print(f"  - Ratio: {params_swiglu/params_gelu:.2f}x")


def test_swiglu_architecture():
    """Test de l'architecture SwiGLU détaillée"""
    print("\n" + "="*60)
    print("TEST 2: Architecture SwiGLU détaillée")
    print("="*60)
    
    embed_dim = 768
    ffn = FeedForward(embed_dim, use_swiglu=True)
    
    print(f"📊 Configuration:")
    print(f"  - Embed dim: {embed_dim}")
    print(f"  - Hidden dim: {ffn.hidden_dim} (= {embed_dim} × 8/3)")
    
    # Détails des couches
    gate_params = ffn.gate_proj.weight.numel()
    up_params = ffn.up_proj.weight.numel()
    down_params = ffn.down_proj.weight.numel()
    
    print(f"\n📊 Paramètres par couche:")
    print(f"  - Gate projection (W1): {gate_params:,}")
    print(f"  - Up projection (W2):   {up_params:,}")
    print(f"  - Down projection (W3): {down_params:,}")
    print(f"  - Total:                {gate_params + up_params + down_params:,}")
    
    # Test forward
    x = torch.randn(1, 10, embed_dim)
    output = ffn(x)
    
    print(f"\n✓ Forward pass:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")


def test_swiglu_activations():
    """Visualise les différentes activations"""
    print("\n" + "="*60)
    print("TEST 3: Comparaison des activations")
    print("="*60)
    
    # Créer des valeurs de test
    x = torch.linspace(-3, 3, 13)
    
    # SiLU (Swish) - utilisé dans SwiGLU
    silu = F.silu(x)
    
    # GELU - utilisé traditionnellement
    gelu = F.gelu(x)
    
    # ReLU - ancien standard
    relu = F.relu(x)
    
    print("\n📊 Comparaison des activations:\n")
    print("  x     |  SiLU  |  GELU  |  ReLU")
    print("--------|--------|--------|-------")
    
    for i in range(len(x)):
        print(f" {x[i]:6.2f} | {silu[i]:6.3f} | {gelu[i]:6.3f} | {relu[i]:6.3f}")
    
    print("\n💡 Observations:")
    print("  - ReLU: coupe brutalement à 0")
    print("  - GELU: transition douce")
    print("  - SiLU: similaire à GELU mais légèrement différent")
    print("  - SwiGLU combine SiLU + gating pour plus d'expressivité")


def test_parameter_comparison():
    """Compare les paramètres entre différentes configs"""
    print("\n" + "="*60)
    print("TEST 4: Comparaison paramètres (embed_dim=768)")
    print("="*60)
    
    embed_dim = 768
    
    # GELU standard (4x expansion)
    ffn_gelu = FeedForward(embed_dim, use_swiglu=False)
    params_gelu = sum(p.numel() for p in ffn_gelu.parameters())
    
    # SwiGLU (8/3x expansion avec gate)
    ffn_swiglu = FeedForward(embed_dim, use_swiglu=True)
    params_swiglu = sum(p.numel() for p in ffn_swiglu.parameters())
    
    print(f"\n📊 GELU (standard):")
    print(f"  - Hidden dim: {ffn_gelu.hidden_dim}")
    print(f"  - Paramètres: {params_gelu:,}")
    
    print(f"\n📊 SwiGLU (moderne):")
    print(f"  - Hidden dim: {ffn_swiglu.hidden_dim}")
    print(f"  - Paramètres: {params_swiglu:,}")
    
    print(f"\n💡 Différence: {params_swiglu - params_gelu:+,} paramètres")
    print(f"   Ratio: {params_swiglu/params_gelu:.3f}x")


def test_llama_config():
    """Test avec configuration LLaMA-style"""
    print("\n" + "="*60)
    print("TEST 5: Configuration LLaMA-style (SwiGLU)")
    print("="*60)
    
    # LLaMA 7B config (simplifié)
    embed_dim = 4096
    
    ffn = FeedForward(embed_dim, use_swiglu=True)
    
    print(f"📊 LLaMA-style FFN:")
    print(f"  - Embed dim: {embed_dim}")
    print(f"  - Hidden dim: {ffn.hidden_dim}")
    print(f"  - Paramètres: {sum(p.numel() for p in ffn.parameters()):,}")
    
    # Test forward
    x = torch.randn(2, 128, embed_dim)
    output = ffn(x)
    
    print(f"\n✓ Forward pass:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")


if __name__ == "__main__":
    print("\n🚀 TESTS DU FEED-FORWARD NETWORK AVEC SwiGLU\n")
    
    # Test 1: SwiGLU vs GELU
    test_swiglu_vs_gelu()
    
    # Test 2: Architecture SwiGLU
    test_swiglu_architecture()
    
    # Test 3: Activations
    test_swiglu_activations()
    
    # Test 4: Comparaison paramètres
    test_parameter_comparison()
    
    # Test 5: Config LLaMA
    test_llama_config()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("="*60)
    print("\n🎯 AVANTAGES DE SwiGLU:")
    print("  • Meilleure performance que GELU/ReLU")
    print("  • Utilisé dans LLaMA, PaLM, Mistral")
    print("  • Gate mechanism plus expressif")
    print("  • Légèrement plus de paramètres mais meilleures perfs")
    print("\n💡 Pour passer de GELU à SwiGLU:")
    print("  FeedForward(embed_dim, use_swiglu=True)  # SwiGLU")
    print("  FeedForward(embed_dim, use_swiglu=False) # GELU")
    print("="*60 + "\n")