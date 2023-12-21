import os

import hydra
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd
from dvc.api import DVCFileSystem
from mlflow.models import infer_signature
from omegaconf import DictConfig
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg["train"]["mlflow_server"])
    mlflow.set_experiment(cfg["train"]["experiment_name"])

    PATH = cfg["data"]["path"]
    if os.path.isfile(f".{PATH}/train.csv"):
        os.remove(f".{PATH}/train.csv")

    fs = DVCFileSystem(f".{PATH}")
    fs.get(f".{PATH}/train.csv", f".{PATH}", recursive=True)
    data = pd.read_csv(f".{PATH}/train.csv")

    columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    X = data[columns]
    y = data.Species
    n_classes = len(np.unique(y))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

    params = cfg["model"]
    lr_model = LogisticRegression(**params)

    lr_model.fit(X_train, y_train)
    y_score = lr_model.predict_proba(X_val)
    y_pred = lr_model.predict(X_val)

    label_binarizer = LabelBinarizer().fit(y_train)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    y_onehot_val = label_binarizer.transform(y_val)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_val[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    plt.legend()

    plt.savefig("roc_curve.png")
    plt.figure(1)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="micro")
    recall = recall_score(y_val, y_pred, average="micro")
    f1 = f1_score(y_val, y_pred, average="micro")
    logloss = log_loss(y_val, y_score)

    # feature importance
    coefficients = lr_model.coef_
    avg_importance = np.mean(np.abs(coefficients), axis=0)
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": avg_importance}
    )
    feature_importance = feature_importance.sort_values("Importance", ascending=True)
    feature_importance.plot(x="Feature", y="Importance", kind="barh", figsize=(10, 6))
    plt.savefig("feature_importance.png")

    plt.figure(2)
    # confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred, normalize="true")

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=["setosa", "versicolor", "virginica"],
    )
    cm_display.plot()
    plt.savefig("val_confusion_matrix.png")

    initial_type = [("float_input", FloatTensorType([None, 4]))]
    options = {id(lr_model): {"zipmap": False}}
    onx_model = convert_sklearn(lr_model, options=options, initial_types=initial_type)

    with open("./model.onnx", "wb") as f:
        f.write(onx_model.SerializeToString())

    onnx_model = onnx.load_model("./model.onnx")
    sess = rt.InferenceSession(
        "./model.onnx",
        providers=rt.get_available_providers(),
    )

    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: np.array(X_train).astype(np.float32)})[0]

    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("val_logloss", logloss)
        mlflow.log_metric("val_roc_auc_macro", roc_auc["macro"])
        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_precision", precision)
        mlflow.log_metric("val_recall", recall)
        mlflow.log_metric("val_f1", f1)

        mlflow.log_artifact("feature_importance.png")
        mlflow.log_artifact("roc_curve.png")
        mlflow.log_artifact("val_confusion_matrix.png")

        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        # Infer the model signature
        signature = infer_signature(X_train, pred_onx)
        mlflow.onnx.save_model(
            onnx_model=onnx_model, path="./model", signature=signature
        )
        mlflow.onnx.log_model(onnx_model, "model", signature=signature)

    filename = f"{cfg['train']['name']}.sav"
    joblib.dump(lr_model, filename)


if __name__ == "__main__":
    train()
