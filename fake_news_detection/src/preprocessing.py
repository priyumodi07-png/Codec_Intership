import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()


try:
    stop_words = set(
        stopwords.words("english")
    )

except:

    nltk.download("stopwords")

    stop_words = set(
        stopwords.words("english")
    )



def clean_text(text):

    text = str(text).lower()


    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )


    text = re.sub(
        "[^a-zA-Z]",
        " ",
        text
    )


    words = text.split()


    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]


    return " ".join(words)