from .preprocessing import clean_text, tokenize
from .oov_detection import detect_oov, count_oov
from .replacement import replace_tokens

def process_text(input_text: str) -> str:
    """
    Process text by replacing OOV words with appropriate IV words.
    
    Args:
        input_text (str): Input text containing potential OOV words
        
    Returns:
        str: Processed text with OOV words replaced by IV words
    """
    # Preprocessing: clean and tokenize the input text
    cleaned = clean_text(input_text)
    tokens = tokenize(cleaned)
    
    # Detect which tokens are OOV
    oov_indices, oov_tokens = detect_oov(tokens)
    
    # Replace OOV words with IV candidates
    replaced_tokens = replace_tokens(tokens, oov_indices)
    
    # Reconstruct text from tokens
    output_text = ' '.join(replaced_tokens)
    return output_text