import pandas as pd
def load_data(file_path: str)->pd.DataFrame:
    """
    Load data from a specified file path.

    Args:
        file_path (str): The path to the data file.
    """
    try:
        data=pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return pd.DataFrame()  
