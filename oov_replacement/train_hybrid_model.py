import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from oov_replacement.replacement import load_or_create_model, VOCABULARY
import pickle

def generate_sample_data(n_samples=1000):
    """
    Generate sample data for training the hybrid model.
    In a real scenario, you would use your actual dataset.
    """
    # Sample some words from vocabulary
    vocab_sample = np.random.choice(VOCABULARY, size=n_samples)
    
    # Create simple contexts (just the word itself for this example)
    contexts = [f"The word is {word}" for word in vocab_sample]
    
    # Labels are indices of words in vocabulary
    labels = [VOCABULARY.index(word) for word in vocab_sample]
    
    return contexts, labels

def main():
    print("Loading or creating hybrid model...")
    load_or_create_model()
    
    from oov_replacement.replacement import hybrid_model, bert_tokenizer, MODEL_PATH
    
    print("Generating sample training data...")
    contexts, labels = generate_sample_data(1000)
    
    # Encode contexts
    encodings = bert_tokenizer(
        contexts,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='tf'
    )
    
    # Convert labels to numpy array
    labels = np.array(labels)
    
    print("Training hybrid model...")
    # Train for just a few steps as an example
    hybrid_model.fit(
        x={
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        },
        y=labels,
        epochs=2,
        batch_size=16,
        verbose=1
    )
    
    print(f"Saving trained model to {MODEL_PATH}...")
    hybrid_model.save(MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    main()