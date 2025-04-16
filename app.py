from flask import Flask, request, render_template, jsonify
from oov_replacement.oov_detection import count_oov
from oov_replacement.evaluation import (
    calculate_recognition_rate,
    calculate_system_replacement_rate,
    calculate_replacement_rate
)
# Import necessary functions for processing
from oov_replacement.preprocessing import clean_text, tokenize
from oov_replacement.oov_detection import detect_oov
from oov_replacement.replacement import replace_tokens

app = Flask(__name__)

# Define the missing process_text function
def process_text(text: str) -> str:
    """
    Cleans, tokenizes, detects OOV, replaces them, and returns the processed text.
    """
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    oov_indices, _ = detect_oov(tokens)
    replaced_tokens = replace_tokens(tokens, oov_indices)
    return " ".join(replaced_tokens)

# Initialize the app without loading the model immediately
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form.get('input_text')

    # Process the text and get OOV count
    try:
        # Now process_text is defined and can be called
        output_text = process_text(input_text)
        # Use the already processed tokens for OOV count if needed,
        # or re-process for simplicity as done here.
        # For efficiency, you might want process_text to return both
        # the replaced text and the OOV count/details.
        oov_count = count_oov(input_text) # count_oov handles its own cleaning/tokenizing

        return jsonify({
            'output_text': output_text,
            'oov_count': oov_count
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'output_text': 'Error processing text',
            'oov_count': 0
        }), 500

@app.route('/calculate_metrics', methods=['POST'])
def calculate_metrics():
    try:
        system_oov_count = int(request.form.get('system_oov_count'))
        manual_oov_count = int(request.form.get('manual_oov_count'))
        correct_conversion_count = int(request.form.get('correct_conversion_count'))
        
        recognition_rate = calculate_recognition_rate(system_oov_count, manual_oov_count)
        system_replacement_rate = calculate_system_replacement_rate(correct_conversion_count, system_oov_count)
        replacement_rate = calculate_replacement_rate(correct_conversion_count, manual_oov_count)
        
        return jsonify({
            'recognition_rate': round(recognition_rate, 2),
            'system_replacement_rate': round(system_replacement_rate, 2),
            'replacement_rate': round(replacement_rate, 2)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'recognition_rate': 0,
            'system_replacement_rate': 0,
            'replacement_rate': 0
        }), 500

if __name__ == '__main__':
    # Set debug=False to avoid loading models twice
    app.run(debug=False)