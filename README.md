# MovieSense 🎬🍿

Analyze movie reviews with ease! ✨ This Streamlit app uses a Bernoulli Naive Bayes model & NLTK to predict sentiment (positive 😊 or negative 😔). Perfect for NLP beginners! #SentimentAnalysis #MachineLearning #Streamlit

---

## Introduction 👋

`MovieSense` is a web application designed to classify the sentiment of movie reviews as either positive or negative. Built with Streamlit for an interactive user interface, it leverages a machine learning model (Bernoulli Naive Bayes) trained on a substantial dataset of movie reviews, incorporating essential Natural Language Processing (NLP) techniques for robust text analysis. This project serves as a practical demonstration of building and deploying a simple yet effective sentiment analysis solution.

---

## Key Aspects 🔑

* **Sentiment Classification:** Predicts whether a movie review expresses a positive or negative sentiment.
* **Web Application:** Interactive user interface built with Streamlit for easy review analysis.
* **Machine Learning Model:** Utilizes a pre-trained Bernoulli Naive Bayes model for sentiment prediction.
* **Natural Language Processing (NLP):** Incorporates essential NLP techniques for text preprocessing.

---

## Key Highlights ✨

* **HTML Tag Removal:** Cleans reviews by stripping away HTML tags using a regular expression.
* **Special Character Handling:** Removes special characters, retaining only alphanumeric content and spaces.
* **Lowercase Conversion:** Standardizes text by converting all characters to lowercase.
* **Stopword Removal:** Eliminates common English stopwords to focus on meaningful terms.
* **Text Stemming:** Reduces words to their root form using Snowball Stemmer for English.
* **Bag of Words (BoW):** Transforms text data into numerical vectors using a `CountVectorizer` with `max_features = 1000` for model input.
* **Pre-trained Model:** Leverages a saved `BernoulliNB` model, chosen for its accuracy, for efficient predictions.
* **User-Friendly Interface:** Intuitive Streamlit UI allows users to paste reviews and get instant sentiment feedback.
* **Model Persistence:** Uses `pickle` to save and load the trained model (`model1.pkl`) and vocabulary (`bow.pkl`) for seamless application use.

---

## How It Works ⚙️

The `MovieSense` application follows a standard machine learning pipeline:

1.  **Data Loading:** The initial dataset (`IMDB-Dataset.csv`) containing movie reviews and their sentiments is loaded.
2.  **Sentiment Encoding:** 'positive' sentiments are converted to `1` and 'negative' to `0`.
3.  **Text Preprocessing:** Raw movie reviews undergo several cleaning steps:
    * HTML tags are removed.
    * Special characters are replaced with spaces.
    * All text is converted to lowercase.
    * Common English stopwords are removed.
    * Words are stemmed to their root form.
4.  **Feature Extraction (Training):** The preprocessed reviews are transformed into numerical vectors using a Bag of Words approach via `CountVectorizer`. This creates a vocabulary of the most frequent words.
5.  **Model Training:** A Bernoulli Naive Bayes classifier is trained on the numerical representation of the reviews and their corresponding sentiments.
6.  **Model Saving:** The trained model and the `CountVectorizer`'s vocabulary (which maps words to their indices) are saved using `pickle`.
7.  **Prediction (Application):**
    * When a user inputs a new review into the Streamlit app, it undergoes the exact same preprocessing steps as the training data.
    * This cleaned review is then converted into a Bag of Words vector using the *saved* vocabulary.
    * Finally, the pre-trained Bernoulli Naive Bayes model predicts the sentiment (positive or negative) based on this vector.

---

## Local Setup & Running the App 🚀

To run `MovieSense` on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/MovieSense.git](https://github.com/YourUsername/MovieSense.git)
    cd MovieSense
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    (You'll need a `requirements.txt` file, which you can generate from `Code.py` and `app.py`. Assuming these are the main dependencies based on the provided code.)
    ```bash
    pip install numpy pandas scikit-learn nltk streamlit
    ```
    *If you encounter NLTK data issues, run:*
    ```python
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
    ```
4.  **Prepare the model and vocabulary:**
    * Ensure you have the `IMDB-Dataset.csv` file in a directory accessible to `Code.py` (e.g., `E:\Datasets\IMDB-Dataset.csv` as per `Code.py`).
    * Run `Code.py` once to train the model and generate `model1.pkl` and `bow.pkl`. These files should be in the same directory as `app.py`.
    ```bash
    python Code.py
    ```
5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser.

---

## Responsibilities Taken 🧑‍💻

* **Data Preprocessing:** Implemented and applied comprehensive text cleaning and preprocessing steps (HTML removal, special character removal, lowercasing, stopword removal, stemming).
* **Model Training & Evaluation:** Trained multiple Naive Bayes classifiers (Gaussian, Multinomial, Bernoulli) and selected the `BernoulliNB` model based on its accuracy for this specific task.
* **Feature Engineering:** Developed the Bag of Words representation using `CountVectorizer` to transform text into numerical input for the model.
* **Model Serialization:** Handled saving the trained model and the `CountVectorizer`'s vocabulary using `pickle` for later use in the web application.
* **Streamlit Application Development:** Designed and implemented the front-end web application using Streamlit for user interaction and prediction display.
* **Deployment Preparation:** Ensured the Streamlit app correctly loads the pre-trained model and vocabulary for seamless inference on new user inputs.

---

## Learning Outcomes & Insights 🧠

* **Practical NLP Pipeline:** Gained hands-on experience in building a complete NLP pipeline from raw text acquisition to sentiment prediction, encompassing data cleaning, feature extraction, model training, and deployment.
* **Text Preprocessing Importance:** Solidified understanding of the critical role of robust text cleaning and normalization (like stemming and stopword removal) in preparing data for machine learning models and significantly impacting their performance.
* **Bag of Words Mechanism:** Deepened understanding of how `CountVectorizer` effectively creates numerical representations from textual data, which is fundamental for many NLP tasks.
* **Naive Bayes in Text Classification:** Explored the application and comparative performance of different Naive Bayes algorithms (Gaussian, Multinomial, Bernoulli) for sentiment analysis, gaining insight into why Bernoulli Naive Bayes is often well-suited for binary presence/absence features derived from text.
* **Streamlit for Rapid Prototyping:** Acquired proficiency in using Streamlit to quickly build and deploy interactive machine learning applications, making models accessible to non-technical users.
* **Model Persistence Best Practices:** Practiced saving and loading machine learning models and associated artifacts (`vocabulary`) using `pickle`, which is essential for deploying trained models without retraining them every time.
* **Reproducibility:** Understood the importance of applying consistent preprocessing steps between the training phase (in `Code.py`) and the inference phase (in `app.py`) to ensure reliable and accurate predictions.

---

## Future Enhancements 📈

* **Expanded Vocabulary:** Experiment with `max_features` in `CountVectorizer` or use TF-IDF for richer feature representation.
* **More Advanced Models:** Integrate and compare performance with other ML models (e.g., Logistic Regression, SVM) or deep learning models (e.g., RNNs, Transformers).
* **Hyperparameter Tuning:** Optimize model parameters using techniques like GridSearchCV or RandomizedSearchCV.
* **User Feedback Loop:** Implement a mechanism for users to provide feedback on prediction accuracy to continuously improve the model.
* **Deployment to Cloud:** Deploy the Streamlit app to a cloud platform (e.g., Hugging Face Spaces, Heroku, AWS).
* **Error Handling & Edge Cases:** Enhance error handling for various user inputs and consider more complex edge cases in text processing.
* **Pre-trained Embeddings:** Explore using pre-trained word embeddings (e.g., Word2Vec, GloVe) for more semantic understanding.

---

## Contributing 🤝

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## Thank You! 🙏

A big thank you to the creators of `numpy`, `pandas`, `scikit-learn`, `nltk`, and `streamlit` for providing such powerful tools. Your contributions make projects like this possible!

Feel free to explore the code, contribute, or provide feedback.
