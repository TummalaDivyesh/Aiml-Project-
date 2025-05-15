from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import logging, traceback
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Create a single Flask app instance and enable CORS
app = Flask(__name__)
CORS(app)  # This will add the necessary CORS headers to all responses

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
try:
    df = pd.read_excel('dataset.xlsx')
    logging.info(f"Dataset loaded successfully. Shape: {df.shape}")

    # Validate required columns
    if 'Student Query' not in df.columns or 'Answer' not in df.columns:
        raise ValueError("Dataset must contain 'Student Query' and 'Answer' columns.")

    # Remove any rows with missing values
    df = df.dropna(subset=['Student Query', 'Answer'])
except Exception as e:
    logging.error(f"Error loading dataset: {str(e)}")
    raise

# Load the Sentence-BERT model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Sentence-BERT model loaded successfully.")

    # Encode all queries in the dataset
    df['Query Embedding'] = df['Student Query'].apply(lambda x: model.encode(x, convert_to_tensor=True))
except Exception as e:
    logging.error(f"Error during model loading or embedding: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()

        if not user_query:
            return jsonify({'error': 'Empty query received'}), 400

        # Encode user query
        user_query_embedding = model.encode(user_query, convert_to_tensor=True)

        # Compute cosine similarity with dataset queries
        similarities = [util.cos_sim(user_query_embedding, query_emb)[0][0].item() for query_emb in df['Query Embedding']]
        
        # Get best matching answer
        best_match_index = similarities.index(max(similarities))
        answer = df.iloc[best_match_index]['Answer']

        return jsonify({'response': answer})
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred while processing your request. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
