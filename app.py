from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import PyPDF2
import re
import nltk
import spacy
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import cohere
from dotenv import load_dotenv

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load environment variables
load_dotenv()

# Initialize Cohere client
try:
    co = cohere.Client(os.getenv('COHERE_API_KEY'))
except Exception as e:
    print(f"Warning: Cohere initialization failed - {e}")
    co = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def clean_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep periods and other essential punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    
    # Remove repeated punctuation
    text = re.sub(r'([.,!?;:])+', r'\1', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)  # Fix missing periods
    text = re.sub(r'(\d)([A-Z])', r'\1. \2', text)     # Fix missing periods after numbers
    
    # Ensure proper sentence breaks
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
    
    # Remove very short lines (likely headers/footers)
    lines = [line for line in text.split('\n') if len(line.split()) > 3]
    
    return ' '.join(lines).strip()

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return clean_text(text)

def analyze_text_with_spacy(text):
    doc = nlp(text)
    
    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Extract main noun phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    # Get important verbs
    main_verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    
    return entities, noun_phrases, main_verbs

def calculate_sentence_importance(sentence, entities, noun_phrases, main_verbs):
    doc = nlp(sentence)
    
    # Check for named entities
    sent_entities = len([ent for ent in doc.ents])
    
    # Check for noun phrases
    sent_nouns = len([chunk for chunk in doc.noun_chunks])
    
    # Check for main verbs
    sent_verbs = len([token for token in doc if token.pos_ == "VERB"])
    
    # Calculate term frequency
    words = word_tokenize(sentence.lower())
    freq_dist = FreqDist(words)
    avg_term_freq = sum(freq_dist.values()) / len(freq_dist) if freq_dist else 0
    
    # Check sentence position (normalized to 0-1)
    pos_score = 1.0  # Will be adjusted in the main loop
    
    # Check sentence length (penalize very short or very long sentences)
    length_score = 1.0
    word_count = len(words)
    if word_count < 5:
        length_score = 0.5
    elif word_count > 40:
        length_score = 0.7
    
    # Calculate final importance score with weighted components
    importance = (
        0.25 * sent_entities +
        0.20 * sent_nouns +
        0.15 * sent_verbs +
        0.20 * avg_term_freq +
        0.10 * pos_score +
        0.10 * length_score
    ) / len(sentence.split())
    
    return importance

def build_similarity_graph(sentences):
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate similarity matrix
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    
    # Create graph
    graph = nx.from_numpy_array(similarity_matrix)
    
    # Calculate centrality scores
    centrality_scores = nx.pagerank(graph)
    
    return centrality_scores

def preprocess_text(text):
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Process each sentence
    processed_sentences = []
    for sentence in sentences:
        # Tokenize and clean words
        words = [word.lower() for word in re.findall(r'\w+', sentence)]
        
        # Remove stopwords
        words = [word for word in words if word not in stop_words]
        
        processed_sentences.append(' '.join(words))
    
    return sentences, processed_sentences

def extract_key_phrases(processed_sentences, original_sentences, top_n=5):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=100)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        
        # Get feature names (phrases)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF scores for each phrase
        avg_scores = tfidf_matrix.mean(axis=0).A1
        
        # Get top phrases
        top_indices = avg_scores.argsort()[-top_n:][::-1]
        key_phrases = [feature_names[i] for i in top_indices]
        
        return key_phrases
    except:
        return []

def format_summary(sentences, key_phrases=None, entities=None):
    formatted_summary = ""
    
    # Add key phrases if available
    if key_phrases:
        formatted_summary += " Key Topics:\n"
        for phrase in key_phrases:
            formatted_summary += f"• {phrase.title()}\n"
    
    # Add named entities if available
    if entities:
        formatted_summary += "\n Key Entities:\n"
        entity_groups = defaultdict(list)
        for entity, label in entities:
            entity_groups[label].append(entity)
        
        for label, items in entity_groups.items():
            formatted_summary += f"• {label}: {', '.join(set(items))}\n"
    
    formatted_summary += "\n Summary Points:\n"
    
    # Add summary points
    for i, sentence in enumerate(sentences):
        if i > 0:
            formatted_summary += "\n"
        formatted_summary += f"• {str(sentence).strip()}"
    
    return formatted_summary

def get_cohere_summary(text, max_tokens=300):
    """Generate a summary using Cohere's API"""
    if not co:
        return "Cohere API not configured. Please check your API key."
    
    try:
        # Generate summary using Cohere
        response = co.summarize(
            text=text,
            length='medium',
            format='paragraph',
            model='command',
            additional_command='Focus on the main points and key findings',
            temperature=0.3
        )
        return response.summary
    except Exception as e:
        print(f"Cohere summarization error: {e}")
        return None

def get_advanced_summary(text, sentences_count=10):
    """Get comprehensive summary using multiple methods"""
    # Clean and preprocess the text
    sentences, processed_sentences = preprocess_text(text)
    
    if len(sentences) <= sentences_count:
        return text
    
    # Get Cohere summary
    cohere_summary = get_cohere_summary(text)
    
    # Get traditional summarization results
    entities, noun_phrases, main_verbs = analyze_text_with_spacy(text)
    key_phrases = extract_key_phrases(processed_sentences, sentences)
    
    # Calculate sentence importance scores
    importance_scores = []
    for i, sentence in enumerate(sentences):
        score = calculate_sentence_importance(sentence, entities, noun_phrases, main_verbs)
        importance_scores.append((i, score))
    
    # Build similarity graph and get centrality scores
    centrality_scores = build_similarity_graph(processed_sentences)
    
    # Combine scores
    final_scores = []
    for i, (_, imp_score) in enumerate(importance_scores):
        cent_score = centrality_scores[i]
        combined_score = 0.6 * imp_score + 0.4 * cent_score
        final_scores.append((i, combined_score))
    
    # Sort sentences by score and select top ones
    top_sentences = sorted(final_scores, key=lambda x: x[1], reverse=True)[:sentences_count]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Restore original order
    
    # Create summary using traditional methods
    traditional_summary = " ".join([sentences[i] for i, _ in top_sentences])
    
    # Format the final summary
    formatted_summary = format_summary(sentences, key_phrases, entities)
    
    # Combine both summaries if Cohere is available
    if cohere_summary:
        final_summary = (
            "AI-Generated Summary (Cohere):\n"
            f"{cohere_summary}\n\n"
            "Traditional Analysis Summary:\n"
            f"{traditional_summary}\n\n"
            f"{formatted_summary}"
        )
    else:
        final_summary = (
            "Summary:\n"
            f"{traditional_summary}\n\n"
            f"{formatted_summary}"
        )
    
    return final_summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            text = extract_text_from_pdf(filepath)
            summary = get_advanced_summary(text)
            os.remove(filepath)  # Clean up the uploaded file
            return jsonify({'summary': summary})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Please upload a PDF file'}), 400

if __name__ == '__main__':
    app.run(debug=True)
