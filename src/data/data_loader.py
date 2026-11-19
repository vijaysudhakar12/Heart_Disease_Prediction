import pandas as pd
from pathlib import Path


def load_data(file_path):
    try:
        file_path = Path(file_path)

        data = pd.read_csv(file_path)

        print("Dataset loaded successfully.")
        print("Dataset Shape:", data.shape)

        return data

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None

    except pd.errors.ParserError:
        print("Error: There was a parsing error while reading the file.")
        return None

    except PermissionError:
        print(f"Error: Permission denied for {file_path}.")
        return None
