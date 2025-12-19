import torch
from transformers import (
    BertTokenizer, BertModel,
    FlavaProcessor, FlavaModel
)
import numpy as np

class TextFeatureExtractor:
    def __init__(self):
        # Initialize BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Initialize FLAVA
        self.flava_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
        self.flava_model = FlavaModel.from_pretrained('facebook/flava-full')
        
        # Set models to evaluation mode
        self.bert_model.eval()
        self.flava_model.eval()
    
    def encode_text_with_bert(self, text):
        """
        Encode text using BERT transformer
        Args:
            text: str or list of str for batch processing
        """
        # Tokenize text with proper padding for variable lengths
        inputs = self.bert_tokenizer(
            text,
            return_tensors='pt',
            padding=True,  # ✓ PADDING ACTIVATED: Pads to longest sequence in batch
            truncation=True,
            max_length=512,
            add_special_tokens=True  # Ensures [CLS] and [SEP] tokens
        )
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        # Extract different types of embeddings
        last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooler_output = outputs.pooler_output  # [batch_size, hidden_size] - FIXED SIZE!
        
        # Get [CLS] token embedding (first token) - FIXED SIZE!
        cls_embedding = last_hidden_states[:, 0, :]
        
        # Mean pooling of all tokens (accounting for padding) - FIXED SIZE!
        attention_mask = inputs['attention_mask']
        # Expand attention mask to match hidden states dimensions
        extended_attention_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_states).float()
        
        # Sum only non-padded tokens
        sum_embeddings = torch.sum(last_hidden_states * extended_attention_mask, dim=1)
        sum_mask = torch.sum(extended_attention_mask, dim=1)
        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        
        return {
            'last_hidden_states': last_hidden_states,  # Variable length - use carefully!
            'pooler_output': pooler_output,            # Fixed size - safe for concatenation
            'cls_embedding': cls_embedding,            # Fixed size - safe for concatenation  
            'mean_embedding': mean_embedding,          # Fixed size - safe for concatenation
            'attention_mask': attention_mask
        }
    
    def get_flava_text_features(self, text):
        """
        Extract text features using FLAVA
        Args:
            text: str or list of str for batch processing
        """
        # Process text with FLAVA processor with padding
        inputs = self.flava_processor(
            text=text,
            return_tensors='pt',
            padding=True,  # ✓ PADDING ACTIVATED: Handles variable lengths
            truncation=True,
            max_length=77  # FLAVA typically uses 77 as max text length
        )
        
        # Get FLAVA text features
        with torch.no_grad():
            outputs = self.flava_model.get_text_features(**inputs)
            
        return outputs
    
    def encode_batch_texts(self, texts, batch_size=32):
        """
        Process multiple texts in batches with proper padding
        Args:
            texts: list of strings
            batch_size: number of texts to process at once
        """
        all_bert_features = []
        all_flava_features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}, texts {i+1}-{min(i+batch_size, len(texts))}")
            
            # BERT encoding with automatic padding within batch
            bert_inputs = self.bert_tokenizer(
                batch_texts,
                return_tensors='pt',
                padding='longest',  # ✓ Pads to longest in THIS batch
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                bert_outputs = self.bert_model(**bert_inputs)
                # Use pooler_output (fixed size) instead of variable-length hidden states
                bert_features = bert_outputs.pooler_output  # Shape: [batch_size, 768]
                all_bert_features.append(bert_features)
            
            # FLAVA encoding with automatic padding within batch
            flava_inputs = self.flava_processor(
                text=batch_texts,
                return_tensors='pt',
                padding='longest',  # ✓ Pads to longest in THIS batch
                truncation=True,
                max_length=77
            )
            
            with torch.no_grad():
                flava_features = self.flava_model.get_text_features(**flava_inputs)
                all_flava_features.append(flava_features)
        
        # Concatenate all batches - now all tensors have same dimensions
        return {
            'bert_features': torch.cat(all_bert_features, dim=0),
            'flava_features': torch.cat(all_flava_features, dim=0)
        }
    
    def demonstrate_padding_strategies(self, texts):
        """
        Demonstrate different padding strategies for variable length texts
        """
        print("=== Padding Strategy Demonstrations ===\n")
        
        # Show text lengths
        text_lengths = [len(self.bert_tokenizer.tokenize(text)) for text in texts]
        print("Text lengths (in tokens):", text_lengths)
        print("Min length:", min(text_lengths))
        print("Max length:", max(text_lengths))
        print()
        
        # Strategy 1: Pad to longest in batch
        print("1. Padding='longest' (recommended for batches)")
        inputs_longest = self.bert_tokenizer(
            texts,
            return_tensors='pt',
            padding='longest',  # ✓ Efficient: only pads to longest in current batch
            truncation=True
        )

        
        return inputs_longest
    def bert_to_flava_pipeline(self, text):
        """
        Complete pipeline: BERT encoding -> FLAVA text features
        Args:
            text: str or list of str
        """
        if isinstance(text, str):
            print(f"Processing text: '{text[:50]}...' " if len(text) > 50 else f"Processing text: '{text}'")
        else:
            print(f"Processing {len(text)} texts in batch")
        
        # Step 1: Encode with BERT
        bert_features = self.encode_text_with_bert(text)
        print(f"BERT encoding complete. Shape: {bert_features['pooler_output'].shape}")
        
        # Step 2: Get FLAVA text features
        flava_features = self.get_flava_text_features(text)
        print(f"FLAVA text features extracted. Shape: {flava_features.shape}")
        
        return {
            'bert_features': bert_features,
            'flava_features': flava_features
        }
    
    def compare_embeddings(self, text):
        """
        Compare BERT and FLAVA text representations
        """
        results = self.bert_to_flava_pipeline(text)
        
        bert_embedding = results['bert_features']['pooler_output']
        flava_embedding = results['flava_features']
        
        print("\nEmbedding Comparison:")
        print(f"BERT embedding shape: {bert_embedding.shape}")
        print(f"FLAVA embedding shape: {flava_embedding.shape}")
        print(f"BERT embedding norm: {torch.norm(bert_embedding).item():.4f}")
        print(f"FLAVA embedding norm: {torch.norm(flava_embedding).item():.4f}")
        
        return results

# Example usage
def main():
    # Initialize the feature extractor
    extractor = TextFeatureExtractor()
    
    # Example texts with VARIABLE LENGTHS
    texts = [
        "Short text.",  # Very short
        "The quick brown fox jumps over the lazy dog.",  # Medium
        "Machine learning is transforming the way we process natural language and understand human communication.",  # Long
        "BERT and FLAVA are both powerful transformer-based models for text understanding, multimodal learning, and representation learning in modern NLP applications."  # Very long
    ]
    
    print("=== BERT to FLAVA Text Feature Extraction with Variable Lengths ===\n")
    
    # Demonstrate padding strategies
    extractor.demonstrate_padding_strategies(texts)
    
    print("\n=== Processing Individual Texts ===")
    for i, text in enumerate(texts, 1):
        print(f"\n--- Example {i} (Length: {len(text)} chars) ---")
        results = extractor.compare_embeddings(text)
    
    print("\n=== Batch Processing with Automatic Padding ===")
    batch_results = extractor.encode_batch_texts(texts, batch_size=2)
    print(f"Batch BERT features shape: {batch_results['bert_features'].shape}")
    print(f"Batch FLAVA features shape: {batch_results['flava_features'].shape}")
    
    # Demonstrate single vs batch processing
    print("\n=== Single vs Batch Processing Comparison ===")
    
    # Single processing
    single_bert_features = []
    for text in texts:
        result = extractor.encode_text_with_bert(text)
        single_bert_features.append(result['pooler_output'])
    single_features = torch.cat(single_bert_features, dim=0)
    
    # Batch processing
    batch_features = extractor.encode_text_with_bert(texts)['pooler_output']
    
    print(f"Single processing result shape: {single_features.shape}")
    print(f"Batch processing result shape: {batch_features.shape}")
    print(f"Results are identical: {torch.allclose(single_features, batch_features)}")

# Alternative approach: Using FLAVA's text encoder directly with padding
def flava_text_only_approach(texts):
    """
    Direct FLAVA text encoding with proper padding for variable lengths
    """
    processor = FlavaProcessor.from_pretrained('facebook/flava-full')
    model = FlavaModel.from_pretrained('facebook/flava-full')
    
    # Process texts with padding
    inputs = processor(
        text=texts, 
        return_tensors='pt',
        padding='longest',  # ✓ PADDING ACTIVATED for variable lengths
        truncation=True,
        max_length=77
    )
    
    # Get text features directly from FLAVA
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        full_outputs = model(**inputs)
        text_embeddings = full_outputs.text_embeddings
        
    return {
        'text_features': text_features,
        'text_embeddings': text_embeddings,
        'attention_mask': inputs.get('attention_mask', None)
    }

if __name__ == "__main__":
    main()
    
    # Demonstrate direct FLAVA approach with variable length texts
    print("\n\n=== Direct FLAVA Text Encoding with Variable Lengths ===")
    sample_texts = [
        "Short.",
        "This is a medium length sample text for FLAVA encoding.",
        "This is a much longer sample text that demonstrates how FLAVA handles variable length inputs with proper padding strategies."
    ]
    
    flava_results = flava_text_only_approach(sample_texts)
    print(f"Direct FLAVA text features shape: {flava_results['text_features'].shape}")
    print(f"Direct FLAVA text embeddings shape: {flava_results['text_embeddings'].shape}")
    
    if flava_results['attention_mask'] is not None:
        print(f"Attention mask shape: {flava_results['attention_mask'].shape}")
        print("Attention mask (showing padding):")
        for i, mask in enumerate(flava_results['attention_mask']):
            active_tokens = mask.sum().item()
            total_tokens = len(mask)
            print(f"  Text {i+1}: {active_tokens}/{total_tokens} active tokens")