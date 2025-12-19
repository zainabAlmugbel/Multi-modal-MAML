import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(SelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Check if embed_dim is divisible by num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len, seq_len]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores (scaled dot-product attention)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Transpose and reshape to original dimensions
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        out = self.out_proj(out)
        
        return out

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            ff_dim: Feed-forward dimension (typically 4x embed_dim)
            dropout: Dropout probability
        """
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward network."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len, seq_len]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state that's not considered a model parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Encoder(nn.Module):
    """Full transformer encoder with multiple layers."""
    
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len=5000, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_layers: Number of encoder layers
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(Encoder, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Create stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            mask: Optional mask tensor [batch_size, seq_len, seq_len]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Convert token indices to embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        return x

# Example usage
def create_encoder_example(batch_size=2, seq_len=10):
    # Model hyperparameters
    vocab_size = 10000
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 6
    
    # Create encoder
    encoder = Encoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers)
    
    # Create sample input (token indices)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create attention mask (optional)
    mask = torch.ones(batch_size, seq_len, seq_len)
    
    # Forward pass
    output = encoder(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return encoder, output

if __name__ == "__main__":
    create_encoder_example()
