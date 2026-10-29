
import optuna
import numpy as np
import mlflow
import mlflow.xgboost

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def objective(
    trial,
    X_train_enc, y_train,
    X_valid_enc, y_valid
):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 300, 2000
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.1, log=True
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1, 10
        ),
        "gamma": trial.suggest_float(
            "gamma", 0, 5
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-5, 10, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-5, 10, log=True
        ),
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse"
    }

    # Un run MLflow par trial
    with mlflow.start_run(
        run_name=f"Trial_{trial.number:03d}",
        nested=True
    ):
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_params(params)
        mlflow.log_param("target_transform", "log1p")

        model = XGBRegressor(**params)

        model.fit(
            X_train_enc,
            np.log1p(y_train)
        )

        pred_log = model.predict(X_valid_enc)

        rmse_log = np.sqrt(
            mean_squared_error(
                np.log1p(y_valid),
                pred_log
            )
        )

        # Métrique utilisée par Optuna
        mlflow.log_metric("rmse_log", rmse_log)

        # Métriques de validation dans l'échelle des prix
        pred_price = np.maximum(np.expm1(pred_log), 0)

        mlflow.log_metrics({
            "rmse": np.sqrt(
                mean_squared_error(y_valid, pred_price)
            ),
            "mae": mean_absolute_error(
                y_valid, pred_price
            ),
            "r2": r2_score(y_valid, pred_price)
        })

        # Stocker aussi le résultat dans Optuna
        trial.set_user_attr("rmse_log", float(rmse_log))

        return rmse_log


def optimize_optuna(
    X_train_enc, y_train,
    X_valid_enc, y_valid,
    X_test_enc, y_test,uri_mloflow,nam_esp
):
    mlflow.set_tracking_uri(
        uri_mloflow
    )
    mlflow.set_experiment(
        nam_esp
    )

    # Run parent : regroupe tous les trials
    with mlflow.start_run(
        run_name="Optuna_Optimization"
    ) as parent_run:

        mlflow.log_param("optimization", "Optuna")
        mlflow.log_param("n_trials", 60)
        mlflow.log_param("target_transform", "log1p")

        study = optuna.create_study(
            direction="minimize",
            study_name="xgboost_ames_housing"
        )

        # Les runs enfants sont créés dans objective()
        study.optimize(
            lambda trial: objective(
                trial,
                X_train_enc, y_train,
                X_valid_enc, y_valid
            ),
            n_trials=60
        )

        print("Meilleurs paramètres :", study.best_params)
        print("Meilleur RMSE log :", study.best_value)

        mlflow.log_metric(
            "best_rmse_log",
            study.best_value
        )
        mlflow.log_params({
            f"best_{k}": v
            for k, v in study.best_params.items()
        })

        # Entraîner le meilleur modèle
        best_params = study.best_params.copy()
        best_params.update({
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        })

        best_model = XGBRegressor(**best_params)

        best_model.fit(
            X_train_enc,
            np.log1p(y_train)
        )

        # Évaluation finale sur le test
        y_pred_log = best_model.predict(X_test_enc)
        y_pred = np.maximum(np.expm1(y_pred_log), 0)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmse_log = np.sqrt(
            mean_squared_error(
                np.log1p(y_test),
                y_pred_log
            )
        )

        print("\n--- XGBoost optimisé ---")
        print(f"RMSE     : {rmse:,.2f}")
        print(f"MAE      : {mae:,.2f}")
        print(f"R²       : {r2:.4f}")
        print(f"RMSE log : {rmse_log:.4f}")

        # Run enfant pour le meilleur modèle
        with mlflow.start_run(
            run_name="Best_XGBoost_Model",
            nested=True
        ):
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "rmse_log": rmse_log
            })

            mlflow.xgboost.log_model(
                xgb_model=best_model,
                name="xgb_model"
            )

        print("\nTous les trials et le modèle final")
        print("ont été enregistrés dans MLflow.")

    return best_model, study
