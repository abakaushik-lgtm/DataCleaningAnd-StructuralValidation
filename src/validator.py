import pandas as pd
import numpy as np
import pandera as pa
import re
from typing import Tuple, List, Dict, Any
from src.cleaner import detect_and_standardize_nulls

def get_pandera_type(type_str: str):
    """
    Maps user-facing type string to Pandera/Python types.
    """
    mapping = {
        "Integer": pa.Int,
        "Float": pa.Float,
        "String": pa.String,
        "Date": pa.DateTime,
        "Boolean": pa.Bool
    }
    return mapping.get(type_str, pa.String)

def validate_schema(df: pd.DataFrame, expected_schema: Dict[str, str], check_order: bool = False) -> Dict[str, Any]:
    """
    Validates df against expected_schema using Pandera.
    expected_schema: dict mapping col_name -> type_str ('Integer', 'Float', 'String', 'Date', 'Boolean')
    Returns a dictionary of results:
      - 'is_valid': bool
      - 'errors': list of error dicts {column, row_index, failure_case, error_type}
      - 'missing_columns': list of strings
      - 'unexpected_columns': list of strings
      - 'order_mismatch': bool
    """
    results = {
        "is_valid": True,
        "errors": [],
        "missing_columns": [],
        "unexpected_columns": [],
        "order_mismatch": False
    }
    
    if df is None or df.empty:
        return results
        
    df_cols = list(df.columns)
    schema_cols = list(expected_schema.keys())
    
    # 1. Missing columns
    missing_cols = [c for c in schema_cols if c not in df_cols]
    results["missing_columns"] = missing_cols
    
    # 2. Unexpected columns
    unexpected_cols = [c for c in df_cols if c not in schema_cols]
    results["unexpected_columns"] = unexpected_cols
    
    # 3. Order validation
    if check_order and len(missing_cols) == 0:
        # Check if the overlap columns are in the same relative order
        overlap_df = [c for c in df_cols if c in expected_schema]
        overlap_schema = [c for c in schema_cols if c in df_cols]
        if overlap_df != overlap_schema:
            results["order_mismatch"] = True
            results["is_valid"] = False
            
    if missing_cols:
        results["is_valid"] = False
        
    # Build pandera schema for columns that actually exist
    columns_to_validate = {col: expected_schema[col] for col in schema_cols if col in df_cols}
    
    # If there are no columns to validate type on, return
    if not columns_to_validate:
        return results
        
    pa_columns = {}
    for col, type_str in columns_to_validate.items():
        # Pandera schemas can be strict or relaxed. We use nullable=True to only check types
        # but allow missing values (missing is handled separately in Cleaning)
        pa_type = get_pandera_type(type_str)
        
        # If type is Date, we coerce it first or validate format
        if type_str == "Date":
            # Date can be string or datetime in pandas. We will try converting to datetime
            # to check validity
            pa_columns[col] = pa.Column(pa_type, nullable=True, coerce=True)
        elif type_str == "Integer":
            # Integer columns containing nulls must be Float in Pandas unless we use Int64
            # We coerce it so float representations (like 1.0) can pass as int if safe
            pa_columns[col] = pa.Column(pa.Int64, nullable=True, coerce=True)
        elif type_str == "Float":
            pa_columns[col] = pa.Column(pa.Float64, nullable=True, coerce=True)
        elif type_str == "Boolean":
            pa_columns[col] = pa.Column(pa.Bool, nullable=True, coerce=False)
        else:
            pa_columns[col] = pa.Column(pa.String, nullable=True, coerce=True)
            
    schema = pa.DataFrameSchema(columns=pa_columns, strict=False)
    
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        results["is_valid"] = False
        
        # Extract failure cases
        # e.failure_cases contains columns: schema_context, column, check, check_number, failure_case, index
        fc = e.failure_cases
        for _, row in fc.iterrows():
            col_name = str(row['column'])
            row_idx = row['index']
            fail_val = row['failure_case']
            
            # Formatting clean output
            if pd.isna(row_idx):
                err_desc = f"Schema constraint failed: {row['check']}"
                row_label = "Schema Level"
            else:
                expected_t = expected_schema.get(col_name, "Unknown")
                err_desc = f"Value '{fail_val}' does not conform to expected type '{expected_t}'"
                row_label = int(row_idx)
                
            results["errors"].append({
                "column": col_name,
                "row_index": row_label,
                "failure_case": str(fail_val),
                "error_type": "Type Mismatch",
                "description": err_desc
            })
            
    return results

def evaluate_custom_rules(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates custom rules against the dataframe.
    rules: list of dicts:
      {
        'column': str,
        'operator': str ('>=', '>', '<=', '<', '==', '!=', 'contains', 'length ==', 'is_email', 'is_not_null'),
        'value': Any
      }
    Returns:
      {
        'is_valid': bool,
        'results': list of dicts summarizing rule status,
        'failed_rows': list of dicts with details of rows that failed rules
      }
    """
    failed_rows = []
    rule_results = []
    overall_valid = True
    
    if df is None or df.empty:
        return {"is_valid": True, "results": [], "failed_rows": []}
        
    for rule_idx, rule in enumerate(rules):
        col = rule.get('column')
        op = rule.get('operator')
        ref_val = rule.get('value')
        
        if col not in df.columns:
            rule_results.append({
                "rule_index": rule_idx,
                "rule": f"{col} {op} {ref_val}",
                "status": "Error",
                "message": f"Column '{col}' not found in dataset"
            })
            overall_valid = False
            continue
            
        series = df[col]
        passed_mask = pd.Series(True, index=df.index)
        
        try:
            # 1. Comparison operations (cast series to numeric if relevant)
            if op in ('>=', '>', '<=', '<'):
                num_series = pd.to_numeric(series, errors='coerce')
                num_ref = float(ref_val)
                # Nulls will evaluate to False for the comparison
                if op == '>=':
                    passed_mask = (num_series >= num_ref) | num_series.isna()
                elif op == '>':
                    passed_mask = (num_series > num_ref) | num_series.isna()
                elif op == '<=':
                    passed_mask = (num_series <= num_ref) | num_series.isna()
                elif op == '<':
                    passed_mask = (num_series < num_ref) | num_series.isna()
                    
            elif op == '==':
                passed_mask = (series.astype(str) == str(ref_val)) | series.isna()
                
            elif op == '!=':
                passed_mask = (series.astype(str) != str(ref_val)) | series.isna()
                
            elif op == 'contains':
                passed_mask = series.astype(str).str.contains(str(ref_val), case=False, na=True) | series.isna()
                
            elif op == 'length ==':
                length = int(ref_val)
                # Clean strings before checking length
                lengths = series.astype(str).apply(lambda x: len(x) if x != 'nan' else np.nan)
                passed_mask = (lengths == length) | series.isna()
                
            elif op == 'is_email':
                email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                # Check formatting of non-null string representations
                def match_email(x):
                    if pd.isna(x) or str(x).lower() == 'nan':
                        return True
                    return bool(re.match(email_regex, str(x).strip()))
                passed_mask = series.apply(match_email)
                
            elif op == 'is_not_null':
                passed_mask = series.notna() & (series.astype(str).str.strip().str.lower() != 'nan') & (series.astype(str).str.strip() != '')
                
            # Count failures
            failures = ~passed_mask
            fail_indices = df.index[failures].tolist()
            fail_count = len(fail_indices)
            
            rule_str = f"'{col}' {op}" + (f" '{ref_val}'" if op not in ('is_email', 'is_not_null') else "")
            
            if fail_count > 0:
                overall_valid = False
                rule_results.append({
                    "rule_index": rule_idx,
                    "rule": rule_str,
                    "status": "Failed",
                    "failures": fail_count,
                    "message": f"Rule failed for {fail_count} rows"
                })
                
                # Add sample failures to failed_rows
                for idx in fail_indices[:200]:  # limit detailed failure reporting
                    failed_rows.append({
                        "rule": rule_str,
                        "column": col,
                        "row_index": idx,
                        "value": str(df.loc[idx, col]),
                        "description": f"Value failed rule: {rule_str}"
                    })
                if fail_count > 200:
                    failed_rows.append({
                        "rule": rule_str,
                        "column": col,
                        "row_index": "...",
                        "value": "...",
                        "description": f"And {fail_count - 200} more rows failed this rule"
                    })
            else:
                rule_results.append({
                    "rule_index": rule_idx,
                    "rule": rule_str,
                    "status": "Passed",
                    "failures": 0,
                    "message": "All non-null records passed"
                })
                
        except Exception as e:
            rule_results.append({
                "rule_index": rule_idx,
                "rule": f"{col} {op} {ref_val}",
                "status": "Error",
                "message": f"Failed to execute rule: {str(e)}"
            })
            overall_valid = False
            
    return {
        "is_valid": overall_valid,
        "results": rule_results,
        "failed_rows": failed_rows
    }

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Detects outliers using the Interquartile Range (IQR) method.
    Returns: (boolean_mask, summary_dict)
    """
    mask = pd.Series(False, index=df.index)
    summary = {"lower_bound": None, "upper_bound": None, "q1": None, "q3": None, "iqr": None, "count": 0}
    
    if df is None or df.empty or col not in df.columns:
        return mask, summary
        
    series = pd.to_numeric(df[col], errors='coerce')
    non_null_series = series.dropna()
    
    if len(non_null_series) < 4:
        return mask, summary
        
    q1 = float(non_null_series.quantile(0.25))
    q3 = float(non_null_series.quantile(0.75))
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (series < lower_bound) | (series > upper_bound)
    # Exclude NaNs from outlier mask
    mask = mask & series.notna()
    
    summary = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "count": int(mask.sum())
    }
    
    return mask, summary

def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Detects outliers using the Z-Score method.
    Returns: (boolean_mask, summary_dict)
    """
    mask = pd.Series(False, index=df.index)
    summary = {"mean": None, "std": None, "count": 0, "threshold": threshold}
    
    if df is None or df.empty or col not in df.columns:
        return mask, summary
        
    series = pd.to_numeric(df[col], errors='coerce')
    non_null_series = series.dropna()
    
    if len(non_null_series) < 3:
        return mask, summary
        
    mean = float(non_null_series.mean())
    std = float(non_null_series.std())
    
    if std == 0:
        return mask, summary
        
    z_scores = (series - mean) / std
    mask = z_scores.abs() > threshold
    mask = mask & series.notna()
    
    summary = {
        "mean": mean,
        "std": std,
        "count": int(mask.sum()),
        "threshold": threshold
    }
    
    return mask, summary

def calculate_data_quality_score(df: pd.DataFrame, expected_schema: Dict[str, str] = None, rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generates data quality metrics and an overall score out of 100.
    Based on dimensions: Completeness, Uniqueness, Consistency, Validity.
    """
    if df is None or df.empty:
        return {"overall_score": 0, "completeness": 0, "uniqueness": 0, "validity": 0, "consistency": 0}
        
    num_rows = len(df)
    num_cols = len(df.columns)
    total_cells = num_rows * num_cols
    
    # 1. Completeness Score (non-null / total cells)
    # Detect standard and string null placeholders
    df_nulls_checked = detect_and_standardize_nulls(df)
    missing_cells = df_nulls_checked.isna().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 100.0
    
    # 2. Uniqueness Score (unique rows / total rows)
    dup_rows = df.duplicated().sum()
    uniqueness = ((num_rows - dup_rows) / num_rows * 100) if num_rows > 0 else 100.0
    
    # 3. Validity Score (schema matching & outlier percentage)
    validity = 100.0
    if expected_schema:
        schema_results = validate_schema(df, expected_schema)
        # Ratio of type mismatch failures to total cell count
        mismatch_count = len(schema_results["errors"])
        if total_cells > 0:
            validity_penalty = (mismatch_count / total_cells) * 100
            validity = max(0.0, 100.0 - validity_penalty)
            
    # 4. Consistency Score (based on rules engine failures and date/phone pattern matches)
    consistency = 100.0
    if rules:
        rule_results = evaluate_custom_rules(df, rules)
        failed_count = len(rule_results["failed_rows"])
        # Remove trailing row count limit dots if present
        if failed_count > 0:
            actual_failed = sum([1 for r in rule_results["results"] for _ in range(r.get("failures", 0))])
            consistency_penalty = (actual_failed / (num_rows * len(rules))) * 100
            consistency = max(0.0, 100.0 - consistency_penalty)
            
    # Overall Score (Weighted combination)
    # If no schema or rules, default weights to Completeness and Uniqueness
    if expected_schema and rules:
        weights = {"completeness": 0.3, "uniqueness": 0.2, "validity": 0.3, "consistency": 0.2}
    elif expected_schema:
        weights = {"completeness": 0.4, "uniqueness": 0.3, "validity": 0.3, "consistency": 0.0}
    elif rules:
        weights = {"completeness": 0.4, "uniqueness": 0.3, "validity": 0.0, "consistency": 0.3}
    else:
        weights = {"completeness": 0.5, "uniqueness": 0.5, "validity": 0.0, "consistency": 0.0}
        
    overall_score = (
        completeness * weights["completeness"] +
        uniqueness * weights["uniqueness"] +
        validity * weights["validity"] +
        consistency * weights["consistency"]
    )
    
    return {
        "overall_score": round(overall_score, 1),
        "completeness": round(completeness, 1),
        "uniqueness": round(uniqueness, 1),
        "validity": round(validity, 1),
        "consistency": round(consistency, 1)
    }
