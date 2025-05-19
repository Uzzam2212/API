from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flasgger import Swagger
import re
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all domains on all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure Swagger
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api/docs"
}

swagger = Swagger(app, config=swagger_config, template={
    "info": {
        "title": "Advanced AI Content Detector API",
        "description": "Enhanced API for detecting AI-generated content using advanced NLP techniques",
        "version": "2.0.0",
        "contact": {
            "email": "your-email@example.com"
        },
    },
    "schemes": ["http", "https"],
})

def get_text_statistics(text):
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        words_no_stop = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Calculate various text statistics
        avg_word_length = np.mean([len(word) for word in words_no_stop]) if words_no_stop else 0
        avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences])
        vocabulary_richness = len(set(words_no_stop)) / len(words_no_stop) if words_no_stop else 0
        
        return {
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'vocabulary_richness': vocabulary_richness,
            'total_sentences': len(sentences),
            'total_words': len(words_no_stop)
        }
    except Exception as e:
        logger.error(f"Error in get_text_statistics: {str(e)}")
        raise

def analyze_text(text):
    try:
        logger.info(f"Analyzing text (first 100 chars): {text[:100]}...")
        
        # Enhanced list of indicators for AI-generated content
        ai_indicators = {
            'formal_transitions': r'\b(?:hereby|furthermore|moreover|thus|therefore|consequently|subsequently)\b',
            'academic_vocab': r'\b(?:utilize|implement|facilitate|leverage|optimize|methodology|paradigm)\b',
            'formal_conclusions': r'\b(?:in conclusion|to summarize|in summary|to conclude|ultimately)\b',
            'tech_terms': r'\b(?:artificial intelligence|machine learning|deep learning|neural network|algorithm|automation)\b',
            'optimization_terms': r'\b(?:optimal|efficiently|effectively|streamline|maximize|minimize)\b',
            'academic_phrases': r'\b(?:according to|based on|with respect to|in accordance with|pursuant to)\b',
            'complex_sentences': r'(?:[^.!?]+(?:[,;]\s*and|,\s*or)[^.!?]+[.!?])',
            'repetitive_patterns': r'(?:\b\w+\b)(?:\s+\w+\s+)+\1\b',
            'passive_voice': r'\b(?:is|are|was|were|been|be|being)\s+\w+ed\b',
            'standardized_phrases': r'\b(?:please note that|it is important to|it should be noted)\b'
        }
        
        # Get text statistics
        stats = get_text_statistics(text)
        logger.debug(f"Text statistics: {stats}")
        
        # Calculate indicator matches
        indicator_scores = {}
        for name, pattern in ai_indicators.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            indicator_scores[name] = min(matches / max(stats['total_sentences'], 1), 1.0)
        
        logger.debug(f"Indicator scores: {indicator_scores}")
        
        # Calculate base probability from indicators
        base_probability = np.mean(list(indicator_scores.values()))
        
        # Adjust probability based on text statistics
        adjustments = {
            'word_length': 0.1 if stats['avg_word_length'] > 6 else 0,
            'sentence_length': 0.1 if stats['avg_sentence_length'] > 20 else 0,
            'vocabulary_richness': 0.1 if stats['vocabulary_richness'] > 0.8 else 0
        }
        
        logger.debug(f"Probability adjustments: {adjustments}")
        
        # Calculate final probability
        final_probability = base_probability + sum(adjustments.values())
        final_probability = min(max(final_probability, 0), 1)
        
        logger.info(f"Analysis complete. Final probability: {final_probability}")
        
        return {
            'probability': final_probability,
            'indicators': indicator_scores,
            'statistics': stats,
            'adjustments': adjustments
        }
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}")
        raise

@app.route('/api/detect', methods=['POST'])
def detect_ai_content():
    try:
        logger.info("Received /api/detect request")
        data = request.get_json()
        
        if not data:
            logger.warning("No JSON data received")
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '')
        
        if not text:
            logger.warning("No text provided in request")
            return jsonify({'error': 'No text provided'}), 400

        logger.info("Analyzing text...")
        analysis = analyze_text(text)
        logger.info("Analysis complete")

        response = {
            'result': {
                'ai_probability': round(analysis['probability'] * 100, 2),
                'is_ai_generated': analysis['probability'] > 0.5,
                'indicators': {k: round(v * 100, 2) for k, v in analysis['indicators'].items()},
                'statistics': {k: round(v, 2) if isinstance(v, float) else v 
                             for k, v in analysis['statistics'].items()},
                'adjustments': {k: round(v * 100, 2) for k, v in analysis['adjustments'].items()}
            }
        }
        
        logger.debug(f"Sending response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detector2')
def detector2():
    return render_template('detector2.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
