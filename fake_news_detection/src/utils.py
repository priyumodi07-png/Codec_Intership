import joblib


def save_object(obj, path):
    """
    Save python object using joblib
    """

    joblib.dump(
        obj,
        path
    )


def load_object(path):
    """
    Load saved object
    """

    return joblib.load(
        path
    )