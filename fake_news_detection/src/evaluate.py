import pandas as pd


from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


from src.utils import load_object



def evaluate():


    df = pd.read_csv(
        "data/processed/cleaned_news.csv"
    )


    df.dropna(
        inplace=True
    )



    X = df[
        "clean_content"
    ]

    y = df[
        "label"
    ]



    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )



    model = load_object(
        "models/fake_news_model.pkl"
    )


    vectorizer = load_object(
        "models/tfidf_vectorizer.pkl"
    )



    X_test_vector = vectorizer.transform(
        X_test
    )


    prediction = model.predict(
        X_test_vector
    )



    print(
        classification_report(
            y_test,
            prediction
        )
    )



    print(
        confusion_matrix(
            y_test,
            prediction
        )
    )




if __name__ == "__main__":

    evaluate()