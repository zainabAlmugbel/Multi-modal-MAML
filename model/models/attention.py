import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossModalAttention2D(nn.Module):
    """
    Cross-modal attention mechanism between text and image features
    that works with 2D tensors [batch_size, feature_dim]
    """
    def __init__(self, text_dim, img_dim, hidden_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        
        # Using 1D convolutions to create "pseudo-sequence" from features
        self.text_to_seq = nn.Conv1d(1, 8, kernel_size=1)  # Creates 8 "sequence elements"
        self.img_to_seq = nn.Conv1d(1, 8, kernel_size=1)   # Creates 8 "sequence elements"
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, text_features, img_features):
        """
        Args:
            text_features: Tensor of shape [batch_size, text_dim]
            img_features: Tensor of shape [batch_size, img_dim]
        Returns:
            attended_features: Text features with image context
        """
        batch_size = text_features.size(0)
        
        # Project to common dimension
        text_proj = self.text_proj(text_features)  # [batch_size, hidden_dim]
        img_proj = self.img_proj(img_features)     # [batch_size, hidden_dim]
        
        # Create "pseudo-sequences" from 2D features
        # First unsqueeze to make it [batch_size, 1, hidden_dim]
        text_seq = text_proj.unsqueeze(1)
        img_seq = img_proj.unsqueeze(1)
        
        # Use 1D conv to expand to [batch_size, 8, hidden_dim]
        text_seq = self.text_to_seq(text_seq)
        img_seq = self.img_to_seq(img_seq)
        
        # Reshape for multi-head attention
        text_seq = text_seq.transpose(0, 1)  # [8, batch_size, hidden_dim]
        img_seq = img_seq.transpose(0, 1)    # [8, batch_size, hidden_dim]
        
        # Text attends to image (cross-attention)
        attended_features, attention_weights = self.attention(
            query=text_seq,
            key=img_seq,
            value=img_seq
        )
        
        # Reshape back to original dimensions
        attended_features = attended_features.transpose(0, 1)  # [batch_size, 8, hidden_dim]
        
        # Aggregate the sequence dimension (mean pooling)
        attended_features = torch.mean(attended_features, dim=1)  # [batch_size, hidden_dim]
        
        return attended_features, attention_weights



