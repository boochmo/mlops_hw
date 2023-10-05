import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train(train):
    columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    train_X = train[columns]
    train_y = train.Species

    model = LogisticRegression()
    model.fit(train_X, train_y)

    return model


if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")
    model = train(train_df)

    filename = "irisclass.sav"
    joblib.dump(model, filename)
