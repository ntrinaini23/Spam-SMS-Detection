import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import clean_text

def load_and_preprocess_data():
    df = pd.read_csv('data/spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    # Encode labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Clean text
    df['cleaned'] = df['message'].apply(clean_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['label'], test_size=0.2, random_state=42
    )
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer
