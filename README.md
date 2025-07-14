# Spam-SMS-Detection
🔍 A machine learning-based SMS spam detector using TF-IDF and Logistic Regression. Built as part of my internship with CodSoft.
# 📩 Spam SMS Detection

A machine learning project that classifies SMS messages as **Spam** or **Ham (Legit)** using Natural Language Processing techniques. This project was developed as **Task 4** of my internship at **CodSoft**.

---

## 📌 Features

- Preprocessing of real-world SMS data (`spam.csv`)
- Text vectorization using **TF-IDF**
- Classification using **Logistic Regression**
- Cleaned and modular code structure
- Command-line interface for real-time predictions
- Achieved over **99% accuracy**

---

## 🧠 Techniques & Libraries Used

- Python
- Scikit-learn
- Pandas
- TF-IDF Vectorizer
- Logistic Regression
- Joblib (for model serialization)

---

## 📂 Project Structure

SpamSMSDetection/
├── data/
│ └── spam.csv
├── models/
│ └── spam_classifier.pkl
├── src/
│ ├── preprocess.py
│ ├── train_model.py
│ ├── utils.py
│ └── evaluate.py
├── main.py
├── predict.py
└── requirements.txt




---

## ⚙️ How to Run

1. **Install dependencies**  
```bash
pip install -r requirements.txt


Train the model

'''bash
python main.py


Run live prediction:

'''bash
python predict.py
