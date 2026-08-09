import joblib
import os
import numpy as np

from src.preprocessing import clean_text


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fake_news_model.pkl"
)


VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)


model = joblib.load(
    MODEL_PATH
)


vectorizer = joblib.load(
    VECTORIZER_PATH
)


def predict_news(news):

    cleaned_news = clean_text(
        news
    )


    transformed_news = vectorizer.transform(
        [
            cleaned_news
        ]
    )


    prediction = model.predict(
        transformed_news
    )[0]


    confidence_score = model.decision_function(
        transformed_news
    )[0]


    confidence = (
        1 /
        (
            1 + np.exp(-abs(confidence_score))
        )
    ) * 100


    return (
        int(prediction),
        round(
            confidence,
            2
        )
    )