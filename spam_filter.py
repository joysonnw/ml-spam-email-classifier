import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

model_path = "spam_model.pkl"
vectorizer_path = "vectorizer.pkl"

def train_model(force_retain = False):
    if not force_retain and os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, None, None
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    X_train, X_test, y_train, y_test, = train_test_split(df['message'], df['label'], test_size = 0.2, random_state = 42)

    vectorizer = TfidfVectorizer(lowercase = True, stop_words = 'english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict = True)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return model, vectorizer, accuracy, report_dict

    
