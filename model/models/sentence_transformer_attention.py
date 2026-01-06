import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torchvision import models
import numpy as np

class TextImageAttentionWithSentenceTransformer(nn.Module):
    """
    A model that handles SentenceTransformer's 2D output for
    cross-modal attention with images
    """
    def __init__(self, text_dim=640, img_dim=640, hidden_dim=640, num_classes=1000):
        super().__init__()
        # Dimensions for SentenceTransformer embeddings and image features
        self.text_dim = text_dim  # Typically 768 or 384 depending on model
        self.img_dim = img_dim // 8    # Depends on your CNN backbone
        self.hidden_dim = hidden_dim
        
        # Projections to common dimension
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.img_proj = nn.Linear(self.img_dim, self.hidden_dim)
        
        # Option 1: Expand SentenceTransformer embeddings to "fake" token sequence
        self.text_expansion = nn.Linear(self.text_dim, self.text_dim * 8)
        #self.image_expansion = nn.Linear(img_dim, img_dim * 8)
        # Option 2: Using sequential processing of embeddings
        self.sequential_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Cross attention
        self.cross_attention = nn.MultiheadAttention(self.hidden_dim, num_heads=8)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_classes)
        )
        
    def expand_to_sequence(self, embeddings, seq_len=8):
        """
        Expands 2D SentenceTransformer embeddings to a 3D sequence
        
        Args:
            embeddings: [batch_size, embed_dim]
            seq_len: desired sequence length
        
        Returns:
            sequence: [batch_size, seq_len, embed_dim]
        """
        batch_size = embeddings.size(0)
        
        # Option 1: Simple repeat
        expanded = embeddings.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Option 2: Create varied token-like representations (more advanced)
        # MLP expansion followed by reshape
        expanded_features = self.text_expansion(embeddings)  # [batch_size, text_dim*8]
        expanded_reshaped = expanded_features.reshape(batch_size, seq_len, self.text_dim)
        
        return expanded_reshaped
    
    def forward_option1(self, text_embeddings, img_features):
        """
        Option 1: Expand text embeddings to create a fake sequence
        
        Args:
            text_embeddings: [batch_size, text_dim] from SentenceTransformer
            img_features: [batch_size, num_regions, img_dim] from vision backbone
            
        Returns:
            dict of model outputs
        """
        batch_size = text_embeddings.size(0)
        
        # Expand text embeddings to sequence
        text_seq = self.expand_to_sequence(text_embeddings)  # [batch_size, seq_len, text_dim]
        
        # Project both to common dimension
        text_proj = self.text_proj(text_seq)  # [batch_size, seq_len, hidden_dim]
        img_proj = self.img_proj(img_features)  # [batch_size, num_regions, hidden_dim]
        
        # Prepare for attention
        text_proj = text_proj.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        img_proj = img_proj.transpose(0, 1)    # [num_regions, batch_size, hidden_dim]
        
        # Cross attention: text attends to image
        attended_text, attention_weights = self.cross_attention(
            query=text_proj,
            key=img_proj,
            value=img_proj
        )
        
        # Back to original shape
        attended_text = attended_text.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Aggregate sequence dimension (mean pooling)
        attended_text = torch.mean(attended_text, dim=1)  # [batch_size, hidden_dim]
        
        # Get mean image features as well
        img_features_pooled = torch.mean(img_features, dim=1)  # [batch_size, img_dim]
        img_features_proj = self.img_proj(img_features_pooled)  # [batch_size, hidden_dim]
        
        # Concatenate and classify
        combined = torch.cat([attended_text, img_features_proj], dim=1)  # [batch_size, hidden_dim*2]
        logits = self.classifier(combined)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "attended_text": attended_text,
            "img_features": img_features_proj
        }
    
    def forward_option2(self, text_embeddings, img_features):
        """
        Option 2: Alternative approach without expanding to sequence
        
        Args:
            text_embeddings: [batch_size, text_dim] from SentenceTransformer
            img_features: [batch_size, img_dim] - assume already pooled to 2D
            
        Returns:
            dict of model outputs
        """
        # Project to common space
        text_proj = self.text_proj(text_embeddings)  # [batch_size, hidden_dim]
        img_proj = self.img_proj(img_features)       # [batch_size, hidden_dim]
        
        # Sequential interaction (instead of attention)
        # This is an alternative when you can't use attention
        text_enhanced = self.sequential_layer(text_proj * img_proj)  # Element-wise multiplication
        
        # Concatenate for classification
        combined = torch.cat([text_enhanced, img_proj], dim=1)  # [batch_size, hidden_dim*2]
        logits = self.classifier(combined)
        
        return {
            "logits": logits,
            "text_enhanced": text_enhanced,
            "img_features": img_proj
        }
    
    def forward(self, text_embeddings, img_features, use_option1=True):
        """
        Main forward method that selects which approach to use
        """
        if use_option1:
            # If image features are 2D, expand them to 3D for cross-attention
            if len(img_features.shape) == 2:
                # Convert [batch_size, img_dim] to [batch_size, num_regions=8, img_dim/8]
                img_dim = img_features.size(1)
                num_regions = 8  # Arbitrary choice for number of regions 8
                region_dim = img_dim // num_regions
                
                # Reshape to create pseudo-regions
                #expanded_features = self.image_expansion(img_features)  # [batch_size, text_dim*8]
                #img_features = expanded_features.reshape(img_features.size(0), num_regions,region_dim)
                img_features = img_features.reshape(img_features.size(0), num_regions, region_dim)
            
            return self.forward_option1(text_embeddings, img_features)
        else:
            # If image features are 3D, pool them to 2D for option 2
            if len(img_features.shape) == 3:
                img_features = torch.mean(img_features, dim=1)  # [batch_size, img_dim]
                
            return self.forward_option2(text_embeddings, img_features)


