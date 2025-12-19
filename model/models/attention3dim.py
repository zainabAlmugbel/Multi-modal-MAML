    ## code for attention 3 dim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism between text and image features
    """
    def __init__(self, text_dim, img_dim, hidden_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, text_features, img_features):
        """
        Args:
            text_features: Tensor of shape [batch_size, seq_len, text_dim]
            img_features: Tensor of shape [batch_size, num_patches, img_dim]
        Returns:
            attended_features: Text features with image context
        """
        # Project to common dimension
        text_proj = self.text_proj(text_features)  # [batch_size, seq_len, hidden_dim]
        img_proj = self.img_proj(img_features)     # [batch_size, num_patches, hidden_dim]
        
        # Reshape for multi-head attention
        batch_size = text_features.size(0)
        text_proj = text_proj.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        img_proj = img_proj.transpose(0, 1)    # [num_patches, batch_size, hidden_dim]
        
        # Text attends to image (cross-attention)
        attended_features, attention_weights = self.attention(
            query=text_proj,
            key=img_proj,
            value=img_proj
        )
        
        # Reshape back to original dimensions
        attended_features = attended_features.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        return attended_features, attention_weights


class TextImageModel(nn.Module):
    """
    Model that combines text and image features using cross-modal attention
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        # Image encoder (ResNet or Vision Transformer)
        self.img_encoder = models.resnet50(pretrained=True)
        self.img_encoder.fc = nn.Identity()  # Remove classification head
        self.img_dim = 2048  # ResNet50 feature dimension
        
        # Text encoder (BERT or similar)
        self.text_encoder_name = "bert-base-uncased"
        self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)
        self.text_dim = 768  # BERT hidden dimension
        
        # Image feature projection
        self.img_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # Adjust spatial dimensions
            nn.Flatten(start_dim=2),        # Flatten spatial dimensions
            nn.Linear(self.img_dim, self.img_dim),
            nn.LayerNorm(self.img_dim)
        )
        
        # Cross-modal attention
        self.cross_attn = CrossModalAttention(self.text_dim, self.img_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images: [batch_size, 3, H, W]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        batch_size = images.size(0)
        
        # Extract text features
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, text_dim]
        
        # Extract image features
        with torch.no_grad():  # Freeze image encoder
            img_features = self.img_encoder.conv1(images)
            img_features = self.img_encoder.bn1(img_features)
            img_features = self.img_encoder.relu(img_features)
            img_features = self.img_encoder.maxpool(img_features)
            
            img_features = self.img_encoder.layer1(img_features)
            img_features = self.img_encoder.layer2(img_features)
            img_features = self.img_encoder.layer3(img_features)
            img_features = self.img_encoder.layer4(img_features)  # [batch_size, 2048, h, w]
        
        # Process image features
        img_features = img_features.reshape(batch_size, self.img_dim, -1)  # [batch_size, 2048, h*w]
        img_features = img_features.permute(0, 2, 1)  # [batch_size, h*w, 2048]
        
        # Apply cross-modal attention
        attended_text, attention_weights = self.cross_attn(text_features, img_features)
        
        # Use [CLS] token for classification (first token)
        cls_token = attended_text[:, 0, :]  # [batch_size, text_dim]
        
        # Classification
        logits = self.classifier(cls_token)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "text_features": text_features,
            "img_features": img_features,
            "attended_text": attended_text
        }


# Example usage
def example_usage():
    # Initialize model
    model = TextImageModel(num_classes=10)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Sample inputs
    images = torch.randn(2, 3, 224, 224)  # 2 images, 3 channels, 224x224 resolution
    texts = ["a photo of a cat", "an image of a dog playing"]
    
    # Tokenize texts
    text_encodings = tokenizer(
        texts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )
    
    # Forward pass
    outputs = model(
        images=images,
        input_ids=text_encodings["input_ids"],
        attention_mask=text_encodings["attention_mask"]
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    
    return outputs


# Visualization function for attention maps
def visualize_attention(image, attention_weights, tokenizer, text):
    """
    Visualize the attention between text tokens and image regions
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    
    # Get tokens from text
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # Get attention weights for a specific layer/head
    # We'll use the average across all heads for simplicity
    attn = attention_weights.mean(dim=1)[0].cpu().detach().numpy()
    
    # Reshape the attention weights to match the image size
    h = w = int(np.sqrt(attn.shape[1]))
    attn_map = attn[:len(tokens), :].reshape(len(tokens), h, w)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show original image
    axes[0].imshow(np.array(image))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show attention maps for selected tokens
    for i, (token, ax) in enumerate(zip(tokens[:5], axes[1:]), 1):
        ax.imshow(attn_map[i])
        ax.set_title(f"Attention: '{token}'")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
