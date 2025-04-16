import nltk
from nltk.corpus import words

# Download necessary NLTK data
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Create a set of known words for faster lookup
VOCABULARY = set(words.words())

def detect_oov(tokens: list) -> tuple:
    """
    Detect which tokens are out-of-vocabulary (OOV).
    
    Args:
        tokens (list): List of tokens to check
        
    Returns:
        tuple: (list of indices of OOV tokens, list of OOV tokens)
    """
    oov_indices = []
    oov_tokens = []
    
    for i, token in enumerate(tokens):
        if token.lower() not in VOCABULARY and token.isalpha():
            oov_indices.append(i)
            oov_tokens.append(token)
    
    return oov_indices, oov_tokens

def count_oov(text: str) -> int:
    """
    Count the number of OOV words in the text.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of OOV words
    """
    from .preprocessing import clean_text, tokenize
    
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    oov_indices, _ = detect_oov(tokens)
    
    return len(oov_indices)