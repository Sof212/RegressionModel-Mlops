
from pathlib import Path
from typing import List

import joblib
import mlflow.xgboost
import numpy as np
import pandas as pd
from pydantic import BaseModel


# Racine du projet : RegressionModel-Mlops/
BASE_DIR = Path(__file__).resolve().parents[2]

ORDINAL_PATH = BASE_DIR / "mlartifacts" / "mlartifactsordinal_encoder.pkl"
ONEHOT_PATH = BASE_DIR / "mlartifacts" / "mlartifactsonehot_encoder.pkl"

# Dossier MLflow contenant MLmodel et model.ubj
MODEL_PATH = (
    BASE_DIR
    / "Notebooks"
    / "mlruns"
    / "1"
    / "models"
    / "m-7ac803a139bd4adaa9c612b4f11e1101"
    / "artifacts"
)


class HouseData(BaseModel):
    ExterQual: str
    KitchenQual: str
    BsmtQual: str
    GarageFinish: str
    Foundation: str
    Neighborhood: str
    CentralAir: str

    GrLivArea: int
    GarageCars: int
    GarageArea: int
    TotalBsmtSF: int
    FirstFlrSF: int
    FullBath: int
    TotRmsAbvGrd: int
    YearBuilt: int
    YearRemodAdd: int
    GarageYrBlt: float
    MasVnrArea: float
    Fireplaces: int


class InferenceRequest(BaseModel):
    data: List[HouseData]


def predict(request: InferenceRequest) -> List[float]:
    # Convertir les entrées en DataFrame
    input_data = pd.DataFrame(
        [item.model_dump() for item in request.data]
    )

    # Adapter le nom à celui utilisé à l'entraînement
    input_data = input_data.rename(
        columns={"FirstFlrSF": "1stFlrSF"}
    )

    # Charger les encodeurs
    ord_enc = joblib.load(ORDINAL_PATH)
    ohe_enc = joblib.load(ONEHOT_PATH)

    ordinal_cols = [
        "ExterQual",
        "KitchenQual",
        "BsmtQual",
        "GarageFinish",
        "CentralAir",
    ]
    nominal_cols = ["Foundation", "Neighborhood"]

    # Appliquer les transformations apprises sur le train
    input_data[ordinal_cols] = ord_enc.transform(
        input_data[ordinal_cols]
    )

    nominal_values = ohe_enc.transform(
        input_data[nominal_cols]
    )

    nominal_df = pd.DataFrame(
        nominal_values,
        columns=ohe_enc.get_feature_names_out(nominal_cols),
        index=input_data.index,
    )

    input_data = input_data.drop(columns=nominal_cols)
    input_data = pd.concat([input_data, nominal_df], axis=1)

    # Charger le modèle MLflow
    model = mlflow.xgboost.load_model(str(MODEL_PATH))

    # Respecter l'ordre des variables attendu par le modèle
    if hasattr(model, "feature_names_in_"):
        input_data = input_data.reindex(
            columns=model.feature_names_in_,
            fill_value=0,
        )

    # Prédire le prix puis inverser log1p
    predictions_log = model.predict(input_data)
    predictions = np.expm1(predictions_log)

    return predictions.tolist()
