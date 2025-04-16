import re
import string
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt')

def clean_text(text: str) -> str:
    """
    Clean the input text by removing special characters and normalizing.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters but keep apostrophes for contractions
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

def tokenize(text: str) -> list:
    """
    Tokenize the input text into words.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of tokens
    """
    # Make sure punkt is downloaded before tokenizing
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt')
    
    # Use simple split as a fallback if word_tokenize still fails
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"Warning: NLTK tokenization failed ({e}), falling back to simple split")
        return text.split()