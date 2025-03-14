# PDF Processing and Summarization Tool

A Flask-based web application that processes PDF documents to generate intelligent summaries using natural language processing and advanced text analysis techniques.

## Features

- PDF text extraction
- Advanced text summarization
- Key topics identification
- Named entity recognition
- Important phrase extraction
- Clean and modern web interface

## Technologies Used

- Python 3.11
- Flask 3.0.0
- spaCy for NLP
- PyPDF2 for PDF processing
- NLTK for text processing
- scikit-learn for text analysis
- NetworkX for text graph analysis
- Sumy for text summarization

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Access the web interface at `http://localhost:5000`
2. Upload a PDF file (max size: 16MB)
3. View the generated summary and analysis

## Project Structure

- `app.py`: Main Flask application and text processing logic
- `templates/`: HTML templates
- `uploads/`: Directory for temporary PDF storage
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (create from .env.example)

## License

MIT License
