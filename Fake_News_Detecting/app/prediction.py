import os
import joblib
try:
    from preprocessing import clean_text
except ImportError:
    from .preprocessing import clean_text


class FakeNewsPredictor:
    def __init__(self):
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        self.model_path = os.path.join(models_dir, 'best_model.pkl')
        self.vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        
        self.load_models()
        
    def load_models(self):
        """Loads the trained model and vectorizer if they exist."""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.is_loaded = True
            except Exception as e:
                print(f"Error loading models: {e}")
                self.is_loaded = False
        else:
            self.is_loaded = False

    def predict(self, text):
        """
        Predicts whether the given text is REAL or FAKE.
        Returns a dictionary with prediction and confidence scores.
        """
        if not self.is_loaded:
            return {"error": "Models not loaded. Please train the model first."}
            
        if not text or not text.strip():
            return {"error": "Empty text provided."}
            
        # Preprocess text
        cleaned = clean_text(text)
        if not cleaned:
            return {"error": "Text too short after cleaning."}
            
        # Vectorize
        vectorized = self.vectorizer.transform([cleaned])
        
        # Predict
        prediction = self.model.predict(vectorized)[0]
        
        # Get probabilities if supported by the model
        confidence = None
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(vectorized)[0]
                # Assuming classes_ are ['FAKE', 'REAL'] or similar
                # We can just take the max probability for the predicted class
                confidence = round(max(proba) * 100, 2)
            elif hasattr(self.model, "decision_function"):
                # If the model uses decision_function, we can't easily extract a strict percentage
                # Let's set it to None to avoid confusing the user
                pass
        except Exception:
            pass
            
        return {
            "prediction": prediction,
            "confidence": confidence,
            "cleaned_text": cleaned
        }

predictor = FakeNewsPredictor()

def predict_news(text):
    if not predictor.is_loaded:
        predictor.load_models()
    return predictor.predict(text)
