from src.preprocess import load_and_preprocess_data
from src.train_model import train_and_save_model
from src.evaluate import evaluate_model

import joblib

def main():
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
    train_and_save_model(X_train, y_train, vectorizer)

    model, _ = joblib.load('models/spam_classifier.pkl')
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
