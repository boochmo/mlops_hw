import os

import hydra
import pandas as pd
import requests
from dvc.api import DVCFileSystem
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    PATH = cfg["data"]["path"]
    if os.path.isfile(f".{PATH}/test.csv"):
        os.remove(f".{PATH}/test.csv")

    fs = DVCFileSystem(f".{PATH}")
    fs.get(f".{PATH}/test.csv", f".{PATH}", recursive=True)
    test = pd.read_csv(f".{PATH}/test.csv")

    columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    X_test = test[columns]

    url = f"{cfg['infer']['mlflow_server']}"

    # Create dictionary with pandas DataFrame in the split orientation
    json_data = {"dataframe_split": X_test.to_dict(orient="split")}

    # Score model
    response = requests.post(url, json=json_data)
    print(f"\nPredictions:\n${response.json()}")


if __name__ == "__main__":
    main()
