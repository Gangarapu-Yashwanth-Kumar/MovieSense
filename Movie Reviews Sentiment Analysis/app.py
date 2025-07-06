import streamlit as st
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import os

# --- Streamlit UI Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")

# Ensure NLTK data is available
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

# --- Text Cleaning Functions (Copied from your original script) ---
def clean(text):
    """Removes HTML tags from the text."""
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

def is_special(text):
    """Removes special characters, keeping only alphanumeric and spaces."""
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def to_lower(text):
    """Converts all text to lowercase."""
    return text.lower()

def rem_stopwords(text):
    """Removes common English stopwords."""
    stop_words = set(stopwords.words('english'))
    # Ensure text is a string before tokenizing
    if not isinstance(text, str):
        text = str(text)
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

def stem_txt(text):
    """Applies stemming to words."""
    ss = SnowballStemmer('english')
    # Ensure text is iterable (list of words)
    if isinstance(text, list):
        return " ".join([ss.stem(w) for w in text])
    else:
        return ss.stem(str(text)) # Handle single word or non-list input


# --- Load Model and Vocabulary ---
@st.cache_resource
def load_model_and_vocabulary():
    """Loads the pre-trained model and vocabulary."""
    try:
        # Assuming model1.pkl and bow.pkl are in the same directory as this script
        model_path = os.path.join(os.path.dirname(__file__), 'model1.pkl')
        vocab_path = os.path.join(os.path.dirname(__file__), 'bow.pkl')

        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(vocab_path, 'rb') as vocab_file:
            word_dict = pickle.load(vocab_file)
        return model, word_dict
    except FileNotFoundError:
        st.error("Error: Model files (model1.pkl or bow.pkl) not found. "
                 "Please ensure they are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or vocabulary: {e}")
        st.stop()

model, word_dict = load_model_and_vocabulary()

# Create a mapping from word to index for efficient lookup
word_to_index = {word: idx for word, idx in word_dict.items()}
vocab_size = len(word_dict)

# --- Streamlit UI ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("""
This app predicts the sentiment (positive or negative) of a movie review.
Enter your review below and click "Analyze"!
""")

# Text input area for the review
user_review = st.text_area("Enter your movie review here:", height=200,
                           placeholder="e.g., This movie was absolutely fantastic, loved every moment!")

if st.button("Analyze Sentiment"):
    if user_review:
        with st.spinner("Analyzing..."):
            # 1. Clean the review
            f1 = clean(user_review)
            f2 = is_special(f1)
            f3 = to_lower(f2)
            f4 = rem_stopwords(f3) # This returns a list of words
            f5 = stem_txt(f4)     # This joins the list into a single string

            # 2. Create Bag of Words vector for the input review
            # Initialize a vector of zeros with the size of the vocabulary
            input_vector = [0] * vocab_size

            # Tokenize the stemmed review string
            words_in_new_review = word_tokenize(f5)

            for word in words_in_new_review:
                if word in word_to_index:
                    input_vector[word_to_index[word]] += 1

            # Reshape for prediction (model expects a 2D array)
            input_vector_np = np.array(input_vector).reshape(1, -1)

            # 3. Make prediction
            try:
                prediction = model.predict(input_vector_np)
                sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😔"

                st.success(f"**Predicted Sentiment:** {sentiment}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter a review to analyze.")

st.markdown("---")
st.markdown("Developed using a Bernoulli Naive Bayes model and NLTK for text preprocessing.")
