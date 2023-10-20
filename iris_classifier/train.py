import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    PATH = cfg["data"]["path"]
    train = pd.read_csv(f".{PATH}/train.csv")

    columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    train_X = train[columns]
    train_y = train.Species

    model = LogisticRegression(
        penalty=cfg["model"]["penalty"],
        tol=cfg["model"]["tol"],
        C=cfg["model"]["C"],
        fit_intercept=cfg["model"]["fit_intercept"],
        max_iter=cfg["model"]["max_iter"],
    )

    model.fit(train_X, train_y)

    filename = f"{cfg['model']['name']}.sav"
    joblib.dump(model, filename)


if __name__ == "__main__":
    train()
