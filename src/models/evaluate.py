from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np



def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur les données de test et retourne les métriques d'évaluation.

    Args:
        model: Le modèle entraîné.
        X_test: Les caractéristiques de test.
        y_test: Les étiquettes de test.

    Returns:
        dict: Un dictionnaire contenant les métriques d'évaluation (MAE, RMSE, R2).
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}