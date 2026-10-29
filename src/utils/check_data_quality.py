
import great_expectations as gx
import pandas as pd

from typing import Tuple, List


def validate_housing_data(
    df: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """
    Validate Ames Housing data with Great Expectations 1.24.1.

    Returns:
        (is_valid, errors)
    """

    print("Starting Ames Housing data validation...")

    # --------------------------------------------------
    # 1. Vérification du schéma
    # --------------------------------------------------

    required_features = [
        "ExterQual",
        "KitchenQual",
        "BsmtQual",
        "GarageFinish",
        "Foundation",
        "Neighborhood",
        "CentralAir",
        "GrLivArea",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "FullBath",
        "TotRmsAbvGrd",
        "YearBuilt",
        "YearRemodAdd",
        "GarageYrBlt",
        "MasVnrArea",
        "Fireplaces",
    ]

    is_training = "SalePrice" in df.columns

    if is_training:
        required_features.append("SalePrice")

    missing_columns = [
        col for col in required_features
        if col not in df.columns
    ]

    if missing_columns:
        errors = [
            f"Missing required column: {col}"
            for col in missing_columns
        ]
        print("Schema validation FAILED.")
        print(errors)
        return False, errors

    print("Schema validation passed.")

    # --------------------------------------------------
    # 2. Initialisation de Great Expectations 1.24.1
    # --------------------------------------------------

    context = gx.get_context(mode="ephemeral")

    datasource = context.data_sources.add_pandas(
        name="housing_datasource"
    )

    data_asset = datasource.add_dataframe_asset(
        name="housing_data"
    )

    batch_definition = (
        data_asset.add_batch_definition_whole_dataframe(
            "housing_batch"
        )
    )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    suite = gx.ExpectationSuite(
        name="housing_validation_suite"
    )

    context.suites.add(suite)

    validator = context.get_validator(
        batch=batch,
        expectation_suite_name=suite.name,
    )

    # --------------------------------------------------
    # 3. Valeurs manquantes
    # --------------------------------------------------

    print("Checking missing values...")

    # Ces deux variables peuvent être manquantes dans
    # le dataset Ames Housing original.
    optional_missing_cols = {
        "GarageYrBlt",
        "MasVnrArea",
    }

    for col in required_features:
        if col not in optional_missing_cols:
            validator.expect_column_values_to_not_be_null(
                column=col
            )

    # --------------------------------------------------
    # 4. Validation des catégories ordinales
    # --------------------------------------------------

    print("Checking categorical values...")

    quality = ["Po", "Fa", "TA", "Gd", "Ex"]

    validator.expect_column_values_to_be_in_set(
        column="ExterQual",
        value_set=quality,
    )

    validator.expect_column_values_to_be_in_set(
        column="KitchenQual",
        value_set=quality,
    )

    validator.expect_column_values_to_be_in_set(
        column="BsmtQual",
        value_set=quality + ["NA"],
    )

    validator.expect_column_values_to_be_in_set(
        column="GarageFinish",
        value_set=["Unf", "RFn", "Fin", "NA"],
    )

    validator.expect_column_values_to_be_in_set(
        column="CentralAir",
        value_set=["N", "Y"],
    )

    validator.expect_column_values_to_be_in_set(
        column="Foundation",
        value_set=[
            "BrkTil",
            "CBlock",
            "PConc",
            "Slab",
            "Stone",
            "Wood",
        ],
    )

    # Les quartiers sont vérifiés par rapport aux
    # catégories présentes dans le jeu de données.
    neighborhoods = df["Neighborhood"].dropna().unique().tolist()

    validator.expect_column_values_to_be_in_set(
        column="Neighborhood",
        value_set=neighborhoods,
    )

    # --------------------------------------------------
    # 5. Validation des colonnes numériques
    # --------------------------------------------------

    print("Checking numeric columns...")

    numeric_cols = [
        "GrLivArea",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "FullBath",
        "TotRmsAbvGrd",
        "YearBuilt",
        "YearRemodAdd",
        "GarageYrBlt",
        "MasVnrArea",
        "Fireplaces",
    ]

    if is_training:
        numeric_cols.append("SalePrice")

    for col in numeric_cols:
        validator.expect_column_values_to_be_of_type(
            column=col,
            type_="int64" if pd.api.types.is_integer_dtype(df[col])
            else "float64" if pd.api.types.is_float_dtype(df[col])
            else "object",
        )

    # --------------------------------------------------
    # 6. Contraintes métier
    # --------------------------------------------------

    print("Checking business constraints...")

    non_negative_cols = [
        "GrLivArea",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "FullBath",
        "TotRmsAbvGrd",
        "MasVnrArea",
        "Fireplaces",
    ]

    for col in non_negative_cols:
        validator.expect_column_values_to_be_between(
            column=col,
            min_value=0,
            mostly=1.0,
        )

    validator.expect_column_values_to_be_between(
        column="YearBuilt",
        min_value=1800,
        max_value=2026,
    )

    validator.expect_column_values_to_be_between(
        column="YearRemodAdd",
        min_value=1800,
        max_value=2026,
    )

    # Une année de rénovation ne devrait pas précéder
    # l'année de construction.
    validator.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="YearRemodAdd",
        column_B="YearBuilt",
        or_equal=True,
    )

    # --------------------------------------------------
    # 7. Validation de SalePrice si disponible
    # --------------------------------------------------

    if is_training:
        validator.expect_column_values_to_be_between(
            column="SalePrice",
            min_value=1,
        )

    # --------------------------------------------------
    # 8. Exécution et résultats
    # --------------------------------------------------

    print("Running validation suite...")

    results = validator.validate()

    failed_expectations = []

    for result in results.results:
        if not result.success:
            config = result.expectation_config
            kwargs = config.kwargs

            expectation_type = config.type
            column = kwargs.get("column", "")

            failed_expectations.append(
                f"{expectation_type} ({column})"
            )

    total_checks = len(results.results)
    passed_checks = sum(
        1 for result in results.results
        if result.success
    )

    if results.success:
        print(
            f"Validation PASSED: "
            f"{passed_checks}/{total_checks} checks successful."
        )
    else:
        print(
            f"Validation FAILED: "
            f"{total_checks - passed_checks}/{total_checks} "
            "checks failed."
        )
        for error in failed_expectations:
            print("-", error)

    return results.success, failed_expectations
