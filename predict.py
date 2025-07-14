import joblib
from src.utils import clean_text

# Load model
try:
    model, vectorizer = joblib.load('models/spam_classifier.pkl')
except FileNotFoundError:
    print("❌ Model not found. Run main.py to train it first.")
    exit()

print("\n📩 Spam SMS Detector")
print("--------------------------")

message = input("Enter your SMS message: ")
cleaned = clean_text(message)
X = vectorizer.transform([cleaned])
pred = model.predict(X)

label = "🚫 Spam" if pred[0] == 1 else "✅ Ham (Legit)"
print(f"\nPrediction: {label}")
