
from pydantic import BaseModel
from typing import List
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost

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
    FirstFlrSF: int  # Alias de la colonne "1stFlrSF"
    FullBath: int
    TotRmsAbvGrd: int
    YearBuilt: int
    YearRemodAdd: int
    GarageYrBlt: float
    MasVnrArea: float
    Fireplaces: int

    model_config = {"populate_by_name": True}


class InferenceRequest(BaseModel):
    data: List[HouseData]


def predict(
    request: InferenceRequest,
    path_ordinal: str,
    path_onehot: str,
) -> List[float]:

    # Convertir les entrées en DataFrame
    input_data = pd.DataFrame(
        [item.model_dump() for item in request.data]
    )

    # Renommer la colonne pour correspondre au dataset original
    input_data = input_data.rename(
        columns={"FirstFlrSF": "1stFlrSF"}
    )

    # Charger le modèle et les encodeurs appris sur le train
    ord_enc = joblib.load(path_ordinal)
    ohe_enc = joblib.load(path_onehot)

    ordinal_cols = [
        "ExterQual",
        "KitchenQual",
        "BsmtQual",
        "GarageFinish",
        "CentralAir",
    ]
    nominal_cols = ["Foundation", "Neighborhood"]

    # Appliquer les encodeurs sans les réentraîner
    input_data[ordinal_cols] = ord_enc.transform(
        input_data[ordinal_cols]
    )

    nominal_values = ohe_enc.transform(input_data[nominal_cols])

    nominal_df = pd.DataFrame(
        nominal_values,
        columns=ohe_enc.get_feature_names_out(nominal_cols),
        index=input_data.index,
    )
    model = mlflow.xgboost.load_model(
    "/Users/sofianeagounikaci/RegressionModel-Mlops/Notebooks/mlruns/1/models/m-7ac803a139bd4adaa9c612b4f11e1101/artifacts"
)

    input_data = input_data.drop(columns=nominal_cols)
    input_data = pd.concat([input_data, nominal_df], axis=1)

    # Respecter l'ordre des colonnes du modèle si disponible
    if hasattr(model, "feature_names_in_"):
        input_data = input_data.reindex(
            columns=model.feature_names_in_,
            fill_value=0,
        )




    predictions_log = model.predict(input_data)
    predictions = np.expm1(predictions_log)

    return predictions.tolist()

