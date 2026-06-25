import pandas as pd
import numpy as np

def infer_column_type(series: pd.Series) -> str:
    """
    Infers the logical data type of a pandas Series.
    Returns: 'Numeric', 'Boolean', 'DateTime', 'Categorical', or 'Text'.
    """
    # Drop nulls for inference
    non_nulls = series.dropna()
    if len(non_nulls) == 0:
        return "Unknown"
        
    # Check if physical type is datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DateTime"
        
    # Check if physical type is boolean
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
        
    # Check if values are boolean-like
    unique_vals = set(non_nulls.unique())
    if unique_vals.issubset({True, False, 0, 1, '0', '1', 'True', 'False', 'true', 'false', 'TRUE', 'FALSE', 'T', 'F', 't', 'f'}):
        # Additional check to avoid treating normal integer series like ids as boolean
        if len(unique_vals) <= 2:
            return "Boolean"

    # Check if numeric
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
        
    # Check if datetime strings (try parsing a small sample)
    if series.dtype == 'object' or isinstance(series.dtype, pd.CategoricalDtype):
        sample = non_nulls.head(100).astype(str)
        try:
            # Check if majority of sample can be parsed as dates
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().sum() > 0.8 * len(sample):
                # Ensure it's not just integers parsed as dates
                if not all(x.isdigit() and len(x) < 4 for x in sample):
                    return "DateTime"
        except:
            pass

    # Categorical if unique values count is small relative to dataset size
    num_unique = series.nunique()
    total_count = len(series)
    if num_unique < 20 or (total_count > 0 and (num_unique / total_count) < 0.05):
        return "Categorical"
        
    return "Text"

def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Profiles the dataframe to generate metadata, statistics, and column info.
    """
    if df is None or df.empty:
        return {
            "num_rows": 0,
            "num_cols": 0,
            "duplicate_rows": 0,
            "duplicate_percentage": 0.0,
            "overall_completeness": 0.0,
            "columns_profile": []
        }
        
    num_rows = len(df)
    num_cols = len(df.columns)
    
    # Duplicate rows (exact match)
    duplicate_rows = df.duplicated().sum()
    duplicate_percentage = (duplicate_rows / num_rows * 100) if num_rows > 0 else 0.0
    
    # Calculate missing cells
    total_cells = num_rows * num_cols
    missing_cells = df.isna().sum().sum()
    # Also count empty strings or whitespace-only strings as missing in object/string columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        # Check for empty strings, 'nan', 'null', 'n/a', 'unknown', etc. (case-insensitive)
        stripped = df[col].astype(str).str.strip().str.lower()
        empty_mask = stripped.isin(['', 'nan', 'null', 'n/a', 'unknown', 'none', '<na>'])
        # Avoid counting NaNs twice
        empty_mask = empty_mask & df[col].notna()
        missing_cells += empty_mask.sum()
        
    non_missing_cells = total_cells - missing_cells
    overall_completeness = (non_missing_cells / total_cells * 100) if total_cells > 0 else 0.0
    
    columns_profile = []
    for col in df.columns:
        col_series = df[col]
        missing_count = col_series.isna().sum()
        
        # Also treat common placeholder strings as missing for individual profile
        if col_series.dtype == 'object' or isinstance(col_series.dtype, pd.CategoricalDtype):
            stripped = col_series.astype(str).str.strip().str.lower()
            empty_mask = stripped.isin(['', 'nan', 'null', 'n/a', 'unknown', 'none', '<na>'])
            empty_mask = empty_mask & col_series.notna()
            missing_count += empty_mask.sum()
            
        missing_pct = (missing_count / num_rows * 100) if num_rows > 0 else 0.0
        unique_count = col_series.nunique()
        unique_pct = (unique_count / num_rows * 100) if num_rows > 0 else 0.0
        
        inferred = infer_column_type(col_series)
        
        # Get stats
        stats = {}
        if inferred == "Numeric":
            numeric_series = pd.to_numeric(col_series, errors='coerce')
            stats = {
                "mean": float(numeric_series.mean()) if not pd.isna(numeric_series.mean()) else None,
                "median": float(numeric_series.median()) if not pd.isna(numeric_series.median()) else None,
                "std": float(numeric_series.std()) if not pd.isna(numeric_series.std()) else None,
                "min": float(numeric_series.min()) if not pd.isna(numeric_series.min()) else None,
                "max": float(numeric_series.max()) if not pd.isna(numeric_series.max()) else None,
            }
        
        # Top 5 value distributions
        top_vals_series = col_series.dropna()
        if len(top_vals_series) > 0:
            top_vals = top_vals_series.value_counts().head(5).to_dict()
            top_values = [{"value": str(k), "count": int(v)} for k, v in top_vals.items()]
        else:
            top_values = []
            
        columns_profile.append({
            "column": col,
            "type": str(col_series.dtype),
            "inferred_type": inferred,
            "missing_count": int(missing_count),
            "missing_percentage": float(missing_pct),
            "unique_count": int(unique_count),
            "unique_percentage": float(unique_pct),
            "stats": stats,
            "top_values": top_values
        })
        
    return {
        "num_rows": int(num_rows),
        "num_cols": int(num_cols),
        "duplicate_rows": int(duplicate_rows),
        "duplicate_percentage": float(duplicate_percentage),
        "overall_completeness": float(overall_completeness),
        "columns_profile": columns_profile
    }
