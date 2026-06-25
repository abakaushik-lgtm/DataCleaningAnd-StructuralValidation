import pytest
import pandas as pd
import numpy as np
from src.cleaner import (
    detect_and_standardize_nulls,
    handle_missing_values,
    remove_duplicates,
    standardize_data,
    parse_phone,
    parse_date
)

def test_detect_and_standardize_nulls():
    df = pd.DataFrame({
        "names": ["John", "  ", "NaN", "null", "N/A", "Unknown", "Valid"]
    })
    cleaned = detect_and_standardize_nulls(df)
    # The placeholders should become np.nan
    assert pd.isna(cleaned.loc[1, "names"])
    assert pd.isna(cleaned.loc[2, "names"])
    assert pd.isna(cleaned.loc[3, "names"])
    assert pd.isna(cleaned.loc[4, "names"])
    assert pd.isna(cleaned.loc[5, "names"])
    assert cleaned.loc[0, "names"] == "John"
    assert cleaned.loc[6, "names"] == "Valid"

def test_handle_missing_values():
    df = pd.DataFrame({
        "age": [20, np.nan, 30, 40, np.nan],
        "city": ["New York", "Boston", np.nan, "Chicago", "Boston"]
    })
    
    # 1. Mean imputation
    cleaned_mean, logs = handle_missing_values(df, {"age": "mean"})
    assert cleaned_mean["age"].iloc[1] == 30.0
    assert cleaned_mean["age"].iloc[4] == 30.0
    assert len(logs) == 1
    
    # 2. Median imputation
    cleaned_med, _ = handle_missing_values(df, {"age": "median"})
    assert cleaned_med["age"].iloc[1] == 30.0
    
    # 3. Mode imputation
    cleaned_mode, _ = handle_missing_values(df, {"city": "mode"})
    assert cleaned_mode["city"].iloc[2] == "Boston"
    
    # 4. Remove rows strategy
    cleaned_remove_rows, _ = handle_missing_values(df, {"age": "remove_rows"})
    assert len(cleaned_remove_rows) == 3
    
    # 5. Remove column strategy
    cleaned_remove_col, _ = handle_missing_values(df, {"age": "remove_column"})
    assert "age" not in cleaned_remove_col.columns

def test_remove_duplicates():
    df = pd.DataFrame({
        "id": [1, 1, 2, 3, 3],
        "val": ["A", "A", "B", "C", "D"]
    })
    
    # Exact duplicates check
    cleaned_all, _ = remove_duplicates(df, subset=None, keep='first')
    assert len(cleaned_all) == 4 # Row 1 duplicate of Row 0
    
    # Subset duplicates check on ID
    cleaned_id, _ = remove_duplicates(df, subset=["id"], keep='first')
    assert len(cleaned_id) == 3 # Rows 1 and 4 removed
    
    cleaned_last, _ = remove_duplicates(df, subset=["id"], keep='last')
    assert cleaned_last["val"].iloc[0] == "A" # keeps second index which is A
    assert cleaned_last["val"].iloc[2] == "D" # keeps latest index which is D

def test_parse_phone():
    assert parse_phone("5551234567") == "+1-555-123-4567"
    assert parse_phone("+1 (555) 123-4567") == "+1-555-123-4567"
    assert parse_phone("15551234567") == "+1-555-123-4567"
    assert pd.isna(parse_phone(np.nan))

def test_parse_date():
    assert parse_date("2024.03.12") == "2024-03-12"
    assert parse_date("12/03/2024") == "2024-12-03" # standard pd.to_datetime parsing
    assert parse_date("03-12-2024") == "2024-03-12"
    assert pd.isna(parse_date(np.nan))

def test_standardize_data():
    df = pd.DataFrame({
        "name": ["john DOE", "  MARY jane  "],
        "phone": ["5551234567", "15557654321"],
        "date": ["2024.03.12", "03-12-2024"],
        "text": ["  hello   world  ", "test  "]
    })
    
    ops = {
        "names": ["name"],
        "phones": ["phone"],
        "dates": ["date"],
        "text_trim": ["text"]
    }
    
    cleaned, logs = standardize_data(df, ops)
    assert cleaned["name"].iloc[0] == "John Doe"
    assert cleaned["name"].iloc[1] == "Mary Jane"
    assert cleaned["phone"].iloc[0] == "+1-555-123-4567"
    assert cleaned["phone"].iloc[1] == "+1-555-765-4321"
    assert cleaned["date"].iloc[0] == "2024-03-12"
    assert cleaned["date"].iloc[1] == "2024-03-12"
    assert cleaned["text"].iloc[0] == "hello world"
    assert cleaned["text"].iloc[1] == "test"
    assert len(logs) == 4
