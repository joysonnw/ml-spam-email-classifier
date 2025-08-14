# ML Spam Email Classifier
A simple Machine Learning project that classifies as **Spam** or **Ham** TF-IDF (Term Frequency - Iverse Document Frequency) and a NAive Bayes model, with a STreamlit web app for easy testing.
---
## How it works
1. Loads and Preprocesses the dataset ('spam.csv')..
2. Converts messages into TF-IDF vectors.
3. Trains a Naive Bayes classifier
4. Streamlit UI lets you enter text and see predictions instantly

## How to run
```bash
pip install pandas joblib scikit-learn streamlit
streamlit run spam_ui.py
```
## Requirements
- pandas
- joblib
- scikit-learn
- streamlit
