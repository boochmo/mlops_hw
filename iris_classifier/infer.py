import joblib
import pandas as pd
from sklearn.metrics import classification_report


def infer(model, test):
    columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    test_X = test[columns]
    test_y = test.Species
    prediction = model.predict(test_X)

    print(classification_report(test_y, prediction))

    return prediction


if __name__ == "__main__":
    filename = "irisclass.sav"
    loaded_model = joblib.load(filename)
    test = pd.read_csv("./data/test.csv")

    preds = pd.DataFrame(infer(loaded_model, test))

    pd.DataFrame(preds).to_csv(
        "prediction.csv", sep=",", header=False, encoding="utf-8"
    )
