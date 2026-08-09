import pandas as pd


from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.svm import LinearSVC


from sklearn.metrics import accuracy_score


from src.utils import save_object



DATA_PATH = "data/processed/cleaned_news.csv"


MODEL_PATH = "models/fake_news_model.pkl"


VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"



def train():

    print(
        "Loading Dataset..."
    )


    df = pd.read_csv(
        DATA_PATH
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



    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )



    vectorizer = TfidfVectorizer(
        max_features=5000
    )



    X_train_vector = vectorizer.fit_transform(
        X_train
    )


    X_test_vector = vectorizer.transform(
        X_test
    )



    model = LinearSVC()


    model.fit(
        X_train_vector,
        y_train
    )


    prediction = model.predict(
        X_test_vector
    )


    accuracy = accuracy_score(
        y_test,
        prediction
    )



    print(
        f"Model Accuracy: {accuracy:.4f}"
    )



    save_object(
        model,
        MODEL_PATH
    )


    save_object(
        vectorizer,
        VECTORIZER_PATH
    )



    print(
        "Model saved successfully!"
    )




if __name__ == "__main__":

    train()