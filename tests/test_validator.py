import pytest
import pandas as pd
import numpy as np
from src.validator import (
    validate_schema,
    evaluate_custom_rules,
    detect_outliers_iqr,
    detect_outliers_zscore,
    calculate_data_quality_score
)

def test_validate_schema():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, "twenty-five", 30], # 'twenty-five' is a type mismatch for Integer
        "is_active": [True, False, "not-a-bool"] # type mismatch for Bool
    })
    
    schema = {
        "id": "Integer",
        "name": "String",
        "age": "Integer",
        "is_active": "Boolean"
    }
    
    res = validate_schema(df, schema)
    
    assert res["is_valid"] is False
    # Check that we captured errors
    assert len(res["errors"]) >= 2
    
    # Check details of errors
    error_cols = [err["column"] for err in res["errors"]]
    assert "age" in error_cols
    assert "is_active" in error_cols

def test_evaluate_custom_rules():
    df = pd.DataFrame({
        "age": [20, 16, 25, np.nan],
        "salary": [5000, -1000, 0, 8000],
        "email": ["john@example.com", "bademail", np.nan, "alice@work.org"],
        "phone": ["1234567890", "123", "0987654321", "nan"]
    })
    
    rules = [
        {"column": "age", "operator": ">=", "value": 18},
        {"column": "salary", "operator": ">", "value": 0},
        {"column": "email", "operator": "is_email", "value": ""},
        {"column": "phone", "operator": "length ==", "value": 10}
    ]
    
    res = evaluate_custom_rules(df, rules)
    assert res["is_valid"] is False
    
    # Breakdown results by rule status
    statuses = {r["rule"]: r["status"] for r in res["results"]}
    
    assert statuses["'age' >= '18'"] == "Failed" # 16 is < 18
    assert statuses["'salary' > '0'"] == "Failed" # -1000 and 0 are <= 0
    assert statuses["'email' is_email"] == "Failed" # bademail is not an email
    assert statuses["'phone' length == '10'"] == "Failed" # '123' has length 3, 'nan' is filtered out or passes?
    
    # Check specific failures counted
    rule_age_fails = next(r for r in res["results"] if "age" in r["rule"])["failures"]
    assert rule_age_fails == 1 # only 16 failed

def test_detect_outliers():
    df = pd.DataFrame({
        "val": [10, 12, 11, 13, 10, 11, 12, 14, 100, 11] # 100 is an outlier
    })
    
    # IQR Method
    iqr_mask, iqr_summary = detect_outliers_iqr(df, "val")
    assert iqr_summary["count"] == 1
    assert iqr_mask.iloc[8] == True
    
    # Z-Score Method
    z_mask, z_summary = detect_outliers_zscore(df, "val", threshold=2.0)
    assert z_summary["count"] == 1
    assert z_mask.iloc[8] == True

def test_calculate_data_quality_score():
    # Perfect dataframe
    df_perf = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })
    
    schema = {
        "id": "Integer",
        "name": "String",
        "age": "Integer"
    }
    
    res_perf = calculate_data_quality_score(df_perf, schema)
    assert res_perf["overall_score"] == 100.0
    assert res_perf["completeness"] == 100.0
    assert res_perf["uniqueness"] == 100.0
    
    # Corrupted dataframe
    df_corp = pd.DataFrame({
        "id": [1, 1, 2, 2], # Duplicate rows
        "name": ["Alice", "Alice", np.nan, "Charlie"], # Missing value
        "age": [25, 25, "thirty", 35] # Schema type mismatch
    })
    
    res_corp = calculate_data_quality_score(df_corp, schema)
    assert res_corp["overall_score"] < 100.0
    assert res_corp["completeness"] < 100.0
    assert res_corp["uniqueness"] < 100.0
    assert res_corp["validity"] < 100.0
