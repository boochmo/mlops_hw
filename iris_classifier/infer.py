import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import classification_report


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def infer(cfg: DictConfig):
    filename = f"{cfg['model']['name']}.sav"
    model = joblib.load(filename)

    PATH = cfg["data"]["path"]
    test = pd.read_csv(f".{PATH}/test.csv")

    columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    test_X = test[columns]
    test_y = test.Species
    preds = model.predict(test_X)

    print(classification_report(test_y, preds))

    preds = pd.DataFrame(preds)
    pd.DataFrame(preds).to_csv(
        cfg["infer"]["save_file"], sep=",", header=False, encoding="utf-8"
    )


if __name__ == "__main__":
    infer()
