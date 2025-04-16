import nltk
from nltk.corpus import words
import difflib
import numpy as np
import os
import pickle

# Download necessary NLTK data
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Create a list of known words for replacement
VOCABULARY = words.words()
VOCABULARY_SET = set(VOCABULARY)

# Create a fallback function for when the hybrid model can't be loaded
def find_closest_word_by_edit_distance(word: str, vocabulary: list = VOCABULARY) -> tuple:
    """
    Find the closest word in the vocabulary to the given OOV word based on edit distance.
    
    Args:
        word (str): OOV word to find a replacement for
        vocabulary (list, optional): List of words to search in. Defaults to VOCABULARY.
        
    Returns:
        tuple: (closest word, confidence score)
    """
    # Use difflib to find the closest match
    closest_matches = difflib.get_close_matches(word.lower(), vocabulary, n=3, cutoff=0.6)
    
    if closest_matches:
        # Calculate a confidence score based on the similarity ratio
        similarity = difflib.SequenceMatcher(None, word.lower(), closest_matches[0]).ratio()
        return closest_matches[0], similarity
    else:
        # If no close match is found, return the original word with low confidence
        return word, 0.0

# Try to import TensorFlow and Transformers with error handling
try:
    # First try to install tf-keras if needed
    import importlib.util
    if importlib.util.find_spec("tf_keras") is None:
        import subprocess
        print("Installing tf-keras package...")
        subprocess.check_call(["pip", "install", "tf-keras"])
    
    import tensorflow as tf
    from tensorflow.keras import layers
    from transformers import TFBertModel, BertTokenizer
    
    # Path to model directory
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Path to save/load the hybrid model
    MODEL_PATH = os.path.join(MODEL_DIR, 'hybrid_model')
    TOKENIZER_PATH = os.path.join(MODEL_DIR, 'bert_tokenizer.pkl')
    
    # BERT Whitening Parameters
    WHITENING_PARAMS_DIR = os.path.join(MODEL_DIR, 'whitening_params')
    os.makedirs(WHITENING_PARAMS_DIR, exist_ok=True)
    BERT_MEAN_PATH = os.path.join(WHITENING_PARAMS_DIR, 'bert_mean.npy')
    BERT_W_PATH = os.path.join(WHITENING_PARAMS_DIR, 'bert_W.npy')

    bert_mean = None
    bert_W = None

    try:
        if os.path.exists(BERT_MEAN_PATH) and os.path.exists(BERT_W_PATH):
            bert_mean = np.load(BERT_MEAN_PATH)
            bert_W = np.load(BERT_W_PATH)
            print("Loaded BERT whitening parameters.")
        else:
            print("BERT whitening parameters not found. Whitening will not be applied.")
    except Exception as e:
        print(f"Error loading BERT whitening parameters: {e}. Whitening will not be applied.")
        bert_mean = None
        bert_W = None
    
    # Global variables for model and tokenizer
    hybrid_model = None
    bert_tokenizer = None
    
    HYBRID_MODEL_AVAILABLE = True
    
    def load_or_create_model():
        """
        Load the hybrid model if it exists, otherwise create a new one.
        """
        global hybrid_model, bert_tokenizer, bert_mean, bert_W
        
        # Try to load the tokenizer
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as f:
                bert_tokenizer = pickle.load(f)
        else:
            # Initialize the BERT tokenizer
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # Save the tokenizer
            with open(TOKENIZER_PATH, 'wb') as f:
                pickle.dump(bert_tokenizer, f)
        
        # Try to load the model
        if os.path.exists(MODEL_PATH):
            try:
                hybrid_model = tf.keras.models.load_model(MODEL_PATH)
                print("Loaded existing hybrid model.")
                return
            except:
                print("Could not load existing model. Creating a new one.")
        
        # Create the hybrid model
        print("Creating new hybrid model...")
        
        # Load the base BERT model
        bert_base = TFBertModel.from_pretrained('bert-base-uncased')
        
        # Define model inputs
        input_ids = layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
        attention_mask_input = layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
        
        # Define a function to call BERT
        def call_bert(inputs):
            input_ids, attention_mask = inputs
            attention_mask = tf.cast(attention_mask, tf.int32)
            bert_outputs = bert_base(input_ids, attention_mask=attention_mask)[0]
            return bert_outputs
        
        # Apply the Lambda layer
        bert_outputs = layers.Lambda(
            call_bert,
            output_shape=lambda input_shapes: (input_shapes[0][0], input_shapes[0][1], 768)
        )([input_ids, attention_mask_input])
        
        # Apply BERT Whitening if parameters are available
        whitened_outputs = bert_outputs  # Default to original if params not loaded
        if bert_mean is not None and bert_W is not None:
            print("Applying BERT whitening layer...")
            # Convert numpy arrays to TensorFlow constants
            tf_bert_mean = tf.constant(bert_mean, dtype=tf.float32)
            tf_bert_W = tf.constant(bert_W, dtype=tf.float32)

            def whiten_layer(embeddings):
                # Center the embeddings
                centered_embeddings = embeddings - tf_bert_mean
                # Apply the transformation matrix
                whitened = tf.linalg.matmul(centered_embeddings, tf_bert_W)
                return whitened

            # Get output dimension from W matrix
            whitening_output_dim = bert_W.shape[1]
            whitened_outputs = layers.Lambda(
                whiten_layer,
                output_shape=lambda input_shape: (input_shape[0], input_shape[1], whitening_output_dim)
            )(bert_outputs)
            print(f"Whitening applied. Output dimension: {whitening_output_dim}")
        else:
            print("Skipping BERT whitening layer (parameters not available).")
        
        # Pass the (potentially whitened) embeddings through a Bidirectional LSTM layer
        lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(whitened_outputs)
        
        # Add a Transformer (MultiHeadAttention) layer
        transformer_out = layers.MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
        # Add a residual connection and layer normalization
        transformer_out = layers.LayerNormalization(epsilon=1e-6)(transformer_out + lstm_out)
        
        # Global pooling and dense layers for final output
        pooling = layers.GlobalAveragePooling1D()(transformer_out)
        dense = layers.Dense(64, activation='relu')(pooling)
        
        # For OOV replacement, we'll use a dense layer with vocabulary size output
        output = layers.Dense(len(VOCABULARY), activation='softmax')(dense)
        
        # Build and compile the hybrid model
        hybrid_model = tf.keras.Model(inputs=[input_ids, attention_mask_input], outputs=output)
        hybrid_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the model
        hybrid_model.save(MODEL_PATH)
        print("Created and saved new hybrid model.")

    def encode_text(text, max_len=128):
        """
        Encode text using the BERT tokenizer.
        """
        global bert_tokenizer
        
        if bert_tokenizer is None:
            load_or_create_model()
        
        return bert_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='tf'
        )

    def find_closest_word_by_hybrid_model(word: str, context: str = "") -> tuple:
        """
        Find the closest word using the hybrid model.
        """
        global hybrid_model
        
        if hybrid_model is None:
            load_or_create_model()
        
        # If no context is provided, use the word itself as context
        if not context:
            context = word
        
        # Encode the context
        encodings = encode_text(context)
        
        # Get model predictions
        predictions = hybrid_model.predict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        })
        
        # Get the index of the word with highest probability
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        # Return the predicted word and confidence
        return VOCABULARY[predicted_idx], float(confidence)

except Exception as e:
    print(f"Could not initialize hybrid model: {e}")
    print("Falling back to edit distance-based replacement only.")
    HYBRID_MODEL_AVAILABLE = False

def find_closest_word_hybrid(word: str, context: str = "") -> str:
    """
    Find the closest word using a hybrid approach combining edit distance and deep learning model.
    
    Args:
        word (str): OOV word to find a replacement for
        context (str): Context in which the word appears
        
    Returns:
        str: Closest word from the vocabulary
    """
    # Get candidate from edit distance method
    edit_candidate, edit_score = find_closest_word_by_edit_distance(word)
    
    # If hybrid model is available, try to use it
    if HYBRID_MODEL_AVAILABLE:
        try:
            model_candidate, model_score = find_closest_word_by_hybrid_model(word, context)
            
            # Choose the candidate with the higher confidence score
            if model_score > edit_score * 1.2:  # Give some preference to the model
                return model_candidate
        except Exception as e:
            print(f"Error using hybrid model: {e}")
    
    # Fall back to edit distance if hybrid model is not available or fails
    return edit_candidate

def replace_tokens(tokens: list, oov_indices: list) -> list:
    """
    Replace OOV tokens with their closest IV matches using a hybrid approach.
    
    Args:
        tokens (list): List of all tokens
        oov_indices (list): Indices of OOV tokens
        
    Returns:
        list: Tokens with OOV words replaced
    """
    replaced_tokens = tokens.copy()
    
    # Create context from tokens
    context = " ".join(tokens)
    
    for idx in oov_indices:
        oov_word = tokens[idx]
        replacement = find_closest_word_hybrid(oov_word, context)
        replaced_tokens[idx] = replacement
    
    return replaced_tokens