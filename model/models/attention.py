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


class Alternative2DAttention(nn.Module):
    """
    Simpler approach: convert 2D tensors to 3D by adding a sequence dimension of length 1
    """
    def __init__(self, text_dim, img_dim, hidden_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, text_features, img_features):
        """
        Args:
            text_features: Tensor of shape [batch_size, text_dim]
            img_features: Tensor of shape [batch_size, img_dim]
        """
        # Project to common dimension
        text_proj = self.text_proj(text_features)  # [batch_size, hidden_dim]
        img_proj = self.img_proj(img_features)     # [batch_size, hidden_dim]
        
        # Add sequence dimension of length 1
        text_seq = text_proj.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        img_seq = img_proj.unsqueeze(1)    # [batch_size, 1, hidden_dim]
        
        # Reshape for multi-head attention
        text_seq = text_seq.transpose(0, 1)  # [1, batch_size, hidden_dim]
        img_seq = img_seq.transpose(0, 1)    # [1, batch_size, hidden_dim]
        
        # Text attends to image (cross-attention)
        attended_features, attention_weights = self.attention(
            query=text_seq,
            key=img_seq,
            value=img_seq
        )
        
        # Reshape back and remove sequence dimension
        attended_features = attended_features.transpose(0, 1)  # [batch_size, 1, hidden_dim]
        attended_features = attended_features.squeeze(1)       # [batch_size, hidden_dim]
        
        return attended_features, attention_weights


class TextImageModel2D(nn.Module):
    """
    Model that combines 2D text and image features using cross-modal attention
    """
    def __init__(self, text_dim=768, img_dim=2048, hidden_dim=512, num_classes=1000):
        super().__init__()
        
        # Feature projections
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attn = Alternative2DAttention(text_dim, img_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, text_features, img_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            img_features: [batch_size, img_dim]
        """
        # Project features
        text_proj = self.text_proj(text_features)  # [batch_size, hidden_dim]
        img_proj = self.img_proj(img_features)     # [batch_size, hidden_dim]
        
        # Apply cross-modal attention
        attended_text, attention_weights = self.cross_attn(text_features, img_features)
        
        # Concatenate original and attended features
        fused = torch.cat((attended_text, img_proj), dim=1)  # [batch_size, hidden_dim*2]
        
        # Classification
        logits = self.classifier(fused)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "text_features": text_features,
            "img_features": img_features,
            "attended_text": attended_text
        }


# Approach 3: Using self-attention on concatenated features
class ConcatenatedAttention(nn.Module):
    """
    Combines text and image features by concatenation and applies self-attention
    """
    def __init__(self, text_dim, img_dim, hidden_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # MLP for further processing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, text_features, img_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            img_features: [batch_size, img_dim]
        """
        batch_size = text_features.size(0)
        
        # Project to common dimension
        text_proj = self.text_proj(text_features)  # [batch_size, hidden_dim]
        img_proj = self.img_proj(img_features)     # [batch_size, hidden_dim]
        
        # Create sequence of [text, image] for each example
        # First add sequence dimension
        text_seq = text_proj.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        img_seq = img_proj.unsqueeze(1)    # [batch_size, 1, hidden_dim]
        
        # Concatenate along sequence dimension
        combined = torch.cat((text_seq, img_seq), dim=1)  # [batch_size, 2, hidden_dim]
        
        # Apply self-attention
        # First transpose for attention
        combined = combined.transpose(0, 1)  # [2, batch_size, hidden_dim]
        check_nan = torch.isnan(combined)
        if check_nan.any():
            print(check_nan)
        # Self-attention
        attended_combined, attention_weights = self.self_attention(
            query=combined,
            key=combined,
            value=combined
        )
        
        # Residual connection and normalization
        attended_combined = attended_combined + combined
        attended_combined = self.norm(attended_combined)

        # Apply MLP
        attended_combined = attended_combined + self.mlp(attended_combined)
        
        # Transpose back
        attended_combined = attended_combined.transpose(0, 1)  # [batch_size, 2, hidden_dim]
        check_nan= torch.isnan(attended_combined)
        if torch.any(check_nan): 
            print("attended_combined: ",attended_combined)        
        # Extract attended text and image representations
        attended_text = attended_combined[:, 0]  # [batch_size, hidden_dim]
        attended_img = attended_combined[:, 1]   # [batch_size, hidden_dim]
        
        return attended_text, attended_img, attention_weights


# Example usage
def example_2d_attention():
    batch_size = 4
    text_dim = 768  # BERT embedding dimension
    img_dim = 2048  # ResNet feature dimension
    hidden_dim = 512
    
    # Create sample inputs
    text_features = torch.randn(batch_size, text_dim)
    img_features = torch.randn(batch_size, img_dim)
    
    # Initialize models
    model1 = CrossModalAttention2D(text_dim, img_dim, hidden_dim)
    model2 = Alternative2DAttention(text_dim, img_dim, hidden_dim)
    model3 = ConcatenatedAttention(text_dim, img_dim, hidden_dim)
    
    # Forward pass
    attended_text1, attn_weights1 = model1(text_features, img_features)
    attended_text2, attn_weights2 = model2(text_features, img_features)
    attended_text3, attended_img3, attn_weights3 = model3(text_features, img_features)
    
    print(f"Model 1 - Attended text shape: {attended_text1.shape}")
    print(f"Model 1 - Attention weights shape: {attn_weights1.shape}")
    
    print(f"Model 2 - Attended text shape: {attended_text2.shape}")
    print(f"Model 2 - Attention weights shape: {attn_weights2.shape}")
    
    print(f"Model 3 - Attended text shape: {attended_text3.shape}")
    print(f"Model 3 - Attended image shape: {attended_img3.shape}")
    print(f"Model 3 - Attention weights shape: {attn_weights3.shape}")
    
    return attended_text1, attended_text2, attended_text3, attended_img3


# Complete end-to-end example with classification
class TextImageClassifier(nn.Module):
    """
    Complete text-image classifier with 2D features
    """
    def __init__(self, text_dim=768, img_dim=2048, hidden_dim=512, num_classes=1000):
        super().__init__()
        self.text_encoder = nn.Linear(text_dim, hidden_dim)  # Placeholder for actual text encoder
        self.img_encoder = nn.Linear(img_dim, hidden_dim)    # Placeholder for actual image encoder
        
        # Cross-modal attention
        self.attention = ConcatenatedAttention(text_dim, img_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, text_input, img_input):
        # Extract features (in a real model, these would be from BERT/ResNet)
        text_features = self.text_encoder(text_input)
        img_features = self.img_encoder(img_input)
        
        # Apply cross-modal attention
        attended_text, attended_img, attention_weights = self.attention(text_input, img_input)
        
        # Concatenate for classification
        fused = torch.cat([attended_text, attended_img], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
        return {
            "logits": logits,
            "text_features": text_features,
            "img_features": img_features,
            "attended_text": attended_text,
            "attended_img": attended_img,
            "attention_weights": attention_weights
        }
    



######new code for attention 



class MultimodalAttention(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim=128):
        super(MultimodalAttention, self).__init__()
        
        # Dimensionality reduction for stability
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Attention layers
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor for attention
        self.scale = np.sqrt(hidden_dim)
        
        # Layer normalization for stability
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, image_features, text_features):
        # Apply projections with gradient clipping
        image_proj = self.image_proj(image_features)
        text_proj = self.text_proj(text_features)
        
        # Combine features (for query creation)
        combined = image_proj + text_proj
        combined = self.layer_norm1(combined)
        
        # Create Q, K, V
        queries = self.query_proj(combined)
        keys = self.key_proj(combined)
        values = self.value_proj(combined)
        
        # Compute attention scores with numerical stability safeguards
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.scale + 1e-8)
        
        # Apply softmax with numerical stability
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Add small epsilon to avoid zero probabilities
        attention_probs = attention_probs + 1e-10
        attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)
        
        # Clamp values to prevent extreme outputs
        attention_probs = torch.clamp(attention_probs, min=1e-7, max=1.0)
        
        # Apply attention
        context = torch.matmul(attention_probs, values)
        
        # Apply normalization for stability
        context = self.layer_norm2(context)
        
        # Final projection
        output = self.out_proj(context)
        
        return output

class MetaLearner(nn.Module):
    def __init__(self, image_dim=2048, text_dim=768, hidden_dim=128):
        super(MetaLearner, self).__init__()
        
        self.attention = MultimodalAttention(image_dim, text_dim, hidden_dim)
        
        # Classifier for 4-way task
        self.classifier = nn.Linear(hidden_dim, 4)
        
        # Initialize with proper scaling
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, support_images, support_texts, support_labels, query_images, query_texts):
        """
        Implementation for 4-way 1-shot meta-learning
        
        Args:
            support_images: [tasks, 4, image_dim] - 4 classes, 1 shot each
            support_texts: [tasks, 4, text_dim]
            support_labels: [tasks, 4]
            query_images: [tasks, query_size, image_dim]
            query_texts: [tasks, query_size, text_dim]
        
        Returns:
            query_logits: [tasks, query_size, 4]
        """
        tasks_num = support_images.shape[0]
        query_size = query_images.shape[1]
        
        # Process support set
        support_features = []
        for task_idx in range(tasks_num):
            # Get attention-processed features for support set
            task_support_features = self.attention(
                support_images[task_idx], 
                support_texts[task_idx]
            )  # [4, hidden_dim]
            support_features.append(task_support_features)
        
        # Process query set and compare with prototypes
        all_task_logits = []
        
        for task_idx in range(tasks_num):
            # Process each query example
            task_query_features = []
            for query_idx in range(query_size):
                query_feature = self.attention(
                    query_images[task_idx, query_idx].unsqueeze(0),
                    query_texts[task_idx, query_idx].unsqueeze(0)
                )  # [1, hidden_dim]
                task_query_features.append(query_feature)
            
            task_query_features = torch.cat(task_query_features, dim=0)  # [query_size, hidden_dim]
            
            # Compute distance to each prototype (support example)
            task_logits = []
            for class_idx in range(4):  # 4-way classification
                class_prototype = support_features[task_idx][class_idx]  # [hidden_dim]
                
                # Calculate cosine similarity with scaling for stability
                similarity = F.cosine_similarity(
                    task_query_features,
                    class_prototype.unsqueeze(0).expand(query_size, -1),
                    dim=1
                ) * 10.0  # Scaling factor
                
                task_logits.append(similarity.unsqueeze(1))
            
            task_logits = torch.cat(task_logits, dim=1)  # [query_size, 4]
            all_task_logits.append(task_logits.unsqueeze(0))
        
        query_logits = torch.cat(all_task_logits, dim=0)  # [tasks, query_size, 4]
        
        return query_logits
    
    def meta_learn(self, support_images, support_texts, support_labels, 
                  query_images, query_texts, query_labels, optimizer):
        """
        Meta-learning training step for 4-way 1-shot tasks
        """
        # Forward pass
        query_logits = self.forward(support_images, support_texts, support_labels, 
                                   query_images, query_texts)
        
        # Compute loss with label smoothing for stability
        loss = F.cross_entropy(
            query_logits.view(-1, 4), 
            query_labels.view(-1),
            label_smoothing=0.1  # Prevents overconfident predictions
        )
        
        # Check for NaN and handle
        if torch.isnan(loss):
            print("NaN loss detected! Skipping update.")
            return float('inf')
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        return loss.item()
    
