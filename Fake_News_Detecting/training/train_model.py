import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.preprocessing import clean_text

def train_and_evaluate():
    print("Loading dataset...")
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'fake_or_real_news.csv')
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Run download_dataset.py first.")
        return

    df = pd.read_csv(dataset_path)
    
    print("Preprocessing data...")
    df = df.dropna(subset=['text', 'label'])
    df = df.drop_duplicates()
    
    # Optional: Take a subset if training takes too long during development, but let's train on all ~6000 rows
    print(f"Dataset shape before cleaning: {df.shape}")
    
    # We clean the text
    df['cleaned_text'] = df['text'].apply(clean_text)

    X = df['cleaned_text']
    y = df['label'].str.upper() # REAL or FAKE

    print("Extracting features (TF-IDF)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Increased max_features for better performance on larger dataset
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training models (including Hyperparameter Tuning for Logistic Regression)...")
    
    # Hyperparameter tuning for Logistic Regression
    lr_params = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs'], 'max_iter': [1000]}
    lr_grid = GridSearchCV(LogisticRegression(), lr_params, cv=3, scoring='accuracy', n_jobs=-1)
    
    print("Running GridSearchCV for Logistic Regression...")
    lr_grid.fit(X_train_tfidf, y_train)
    print(f"Best LR Params: {lr_grid.best_params_}")

    models = {
        'Tuned Logistic Regression': lr_grid.best_estimator_,
        'Naive Bayes': MultinomialNB(alpha=0.1), # Alpha tuning
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'Passive Aggressive': PassiveAggressiveClassifier(max_iter=50, random_state=42)
    }

    results = {}
    best_accuracy = 0
    best_model_name = ""
    best_model = None

    for name, model in models.items():
        print(f"Evaluating {name}...")
        if name != 'Tuned Logistic Regression':
            model.fit(X_train_tfidf, y_train)
            
        y_pred = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label='REAL', zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label='REAL', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label='REAL', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']) 

        results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # Save best model and vectorizer
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    
    # Save results as JSON
    results['best_model'] = best_model_name
    with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Training complete. Models and metrics saved in models/ directory.")

if __name__ == "__main__":
    train_and_evaluate()
