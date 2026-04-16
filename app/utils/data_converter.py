import pandas as pd
import numpy as np

def clean_and_convert_data(input_path, output_path):
    """
    Utility script to convert raw datasets into the binary (0/1) format
    required by the SQA Neutrosophic Engine.
    """
    print(f"Reading {input_path}...")
    # Attempt to read CSV, handling different delimiters
    try:
        df = pd.read_csv(input_path)
    except:
        df = pd.read_csv(input_path, sep=';')

    print("Original Columns:", df.columns.tolist())
    
    # --- CUSTOMIZE YOUR MAPPING RULES HERE ---
    
    # Rule 1: Convert Text "Yes"/"No" to 1/0
    df = df.replace({'Yes': 1, 'No': 0, 'Si': 1, 'Sí': 1, 'No': 0})
    df = df.replace({'True': 1, 'False': 0})
    
    # Rule 2: Convert Likert Scales (1-5) to Binary
    # Assumption: 4 and 5 are "Positive" (1), others are "Negative" (0)
    # numeric_cols = df.select_dtypes(include=np.number).columns
    # for col in numeric_cols:
    #     if df[col].max() <= 5: # Likely a Likert scale
    #         df[col] = df[col].apply(lambda x: 1 if x >= 4 else 0)

    # Rule 3: Handle Categorical Variables via One-Hot Encoding
    # Useful if you have a column like "Marital Status" -> [Single, Married, Divorced]
    # This will create columns: "Marital Status_Single", "Marital Status_Married", etc.
    # df = pd.get_dummies(df, drop_first=False) # Convert object/category columns
    
    # ----------------------------------------

    # Ensure all data is numeric 0/1 allowed for the engine
    # Fill NA with 0
    df = df.fillna(0)
    
    print("Converted Data Sample:")
    print(df.head())
    
    df.to_csv(output_path, index=False)
    print(f"Saved converted dataset to {output_path}")

if __name__ == "__main__":
    # Example usage
    # Change "input_data.csv" to your filename
    # clean_and_convert_data("raw_data.csv", "ready_for_sqa.csv")
    print("Please edit this script to point to your raw file and define mapping rules.")
