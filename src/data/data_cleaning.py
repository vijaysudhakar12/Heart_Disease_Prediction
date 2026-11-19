
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_heart.csv"


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return only columns that exist in the DataFrame."""
    return [column for column in columns if column in df.columns]


def clean_data(
    df: pd.DataFrame,
    drop_columns: Optional[Iterable[str]] = None,
    zero_as_missing: Optional[dict[str, list[float]]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clean the Heart Disease dataset.

    Cleaning sequence:
      1. Drop ca and thal (as in notebook 03).
      2. Convert specified invalid zero values to NaN (cholesterol:
         identified during notebook 04 analysis).
      3. Fill categorical/binary missing values with mode.
      4. Fill selected numerical missing values with median or mean.
      5. Convert selected columns to integer/boolean types.
      6. Report remaining missing values and duplicate records.

    Note:
      - The target column 'num' is not imputed.
      - Duplicate rows are reported, not automatically removed.
      - IQR outlier removal is not performed here.
    """
    cleaned = df.copy()

    if drop_columns is None:
        drop_columns = ["ca", "thal"]

    # 1. Drop columns selected in notebook 03.
    columns_to_drop = _existing_columns(cleaned, drop_columns)
    if columns_to_drop:
        cleaned = cleaned.drop(columns=columns_to_drop)

    # 2. Handle known invalid values before imputing missing data.
    # Notebook 04 identified chol == 0 as unavailable cholesterol data.
    if zero_as_missing is None:
        zero_as_missing = {"chol": [0]}

    for column, invalid_values in zero_as_missing.items():
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(invalid_values, pd.NA)

    # 3. Mode imputation for categorical/binary columns.
    mode_columns = ["exang", "slope", "fbs", "restecg"]

    for column in _existing_columns(cleaned, mode_columns):
        modes = cleaned[column].mode(dropna=True)
        if not modes.empty:
            cleaned[column] = cleaned[column].fillna(modes.iloc[0])

    # 4. Median imputation for selected numerical columns.
    median_columns = ["trestbps", "oldpeak"]

    for column in _existing_columns(cleaned, median_columns):
        median_value = cleaned[column].median()
        if pd.notna(median_value):
            cleaned[column] = cleaned[column].fillna(median_value)

    # 5. Mean imputation for selected numerical columns.
    mean_columns = ["chol", "thalch"]

    for column in _existing_columns(cleaned, mean_columns):
        mean_value = cleaned[column].mean()
        if pd.notna(mean_value):
            cleaned[column] = cleaned[column].fillna(mean_value)

    # 6. Convert columns to integer where appropriate.
    # Use nullable checks so unexpected remaining NaNs do not cause
    # an opaque conversion error.
    integer_columns = ["thalch", "trestbps", "chol"]

    for column in _existing_columns(cleaned, integer_columns):
        if cleaned[column].isna().any():
            raise ValueError(
                f"Column '{column}' still contains missing values; "
                "cannot safely convert it to integer."
            )
        cleaned[column] = cleaned[column].round().astype("int64")

    # 7. Convert binary columns to boolean.
    boolean_columns = ["fbs", "exang"]

    for column in _existing_columns(cleaned, boolean_columns):
        if cleaned[column].isna().any():
            raise ValueError(
                f"Column '{column}' still contains missing values; "
                "cannot safely convert it to boolean."
            )
        cleaned[column] = cleaned[column].astype(bool)

    if verbose:
        print("Data cleaning completed.")
        print("\nDataset shape:", cleaned.shape)

        print("\nMissing-value summary:")
        missing_summary = pd.DataFrame({
            "Missing_Count": cleaned.isna().sum(),
            "Missing_Percentage": cleaned.isna().mean() * 100,
        })
        print(missing_summary)

        print("\nDuplicate rows:", cleaned.duplicated().sum())

        if "id" in cleaned.columns:
            print("Duplicate IDs:", cleaned["id"].duplicated().sum())

        if "num" in cleaned.columns:
            print("\nTarget distribution:")
            print(cleaned["num"].value_counts(dropna=False).sort_index())
            print("Target missing values:", cleaned["num"].isna().sum())

    return cleaned


def load_and_clean_data(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    verbose: bool = True, ) -> pd.DataFrame:
    """Load the raw CSV, clean it, save the cleaned CSV, and return it."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset was not found: {input_path}\n"
            "Check the file name and data/raw directory."
        )

    df = pd.read_csv(input_path)
    cleaned_df = clean_data(df, verbose=verbose)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    if verbose:
        print("\nCleaned dataset saved successfully:")
        print(output_path.resolve())
        print("File size:", output_path.stat().st_size, "bytes")

    return cleaned_df


if __name__ == "__main__":
    load_and_clean_data()
