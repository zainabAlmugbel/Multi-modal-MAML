import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionForTwoTexts(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads=10):
        super(SelfAttentionForTwoTexts, self).__init__()
        
        # Embedding layer (you might use pre-trained embeddings instead)
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Processing layers
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, text1_emb, text2_emb, text1_mask=None, text2_mask=None):
        # Get embeddings
        #text1_emb = self.embedding(text1_ids)
        #text2_emb = self.embedding(text2_ids)
        
        tex1_dim= text1_emb.size(0)
        tex2_dim= text2_emb.size(0)
        # Concatenate the two texts
        #combined_emb = torch.cat([text1_emb.mean(dim=0, keepdim=True), text2_emb.mean(dim=0, keepdim=True)], dim=1)
        combined_emb = torch.cat([text1_emb, text2_emb], dim=0)
        #print(combined_emb.shape)
        # Create combined attention mask
        if text1_mask is not None and text2_mask is not None:
            combined_mask = torch.cat([text1_mask, text2_mask], dim=1)
        else:
            combined_mask = None
        
        # Self-attention
        attn_output, attn_weights = self.self_attention(
            query=combined_emb,
            key=combined_emb,
            value=combined_emb,
            key_padding_mask=~combined_mask if combined_mask is not None else None
        )
        
        # Add & norm (residual connection)
        residual = combined_emb + attn_output
        normalized = self.norm1(residual)
        
        # Feed forward network
        ff_output = self.linear2(F.relu(self.linear1(normalized)))
        
        # Final residual connection and normalization
        output = self.norm2(normalized + ff_output)
        
        # Split back into text1 and text2 representations
        text1_len = text1_emb.size(0) * text1_emb.size(1)
        processed_text1 = output[:text1_emb.size(0), :]
        processed_text2 = output[text1_emb.size(0):, :]
        #print(processed_text1.shape, processed_text2.shape)
        return processed_text1, processed_text2, attn_weights
