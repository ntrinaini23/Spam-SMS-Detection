from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_and_save_model(X_train, y_train, vectorizer):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump((model, vectorizer), 'models/spam_classifier.pkl')
