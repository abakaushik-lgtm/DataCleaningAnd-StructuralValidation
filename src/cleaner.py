import pandas as pd
import numpy as np
import re

def detect_and_standardize_nulls(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Converts common missing value placeholders (e.g. '', 'nan', 'null', 'n/a', 'unknown')
    into actual np.nan values so they can be handled by standard imputation strategies.
    """
    df_copy = df.copy()
    cols_to_check = columns if columns is not None else df_copy.columns
    
    placeholders = {'', 'nan', 'null', 'n/a', 'unknown', 'none', '<na>', 'undefined', 'null-value'}
    
    for col in cols_to_check:
        if (df_copy[col].dtype == 'object' or 
            df_copy[col].dtype == 'str' or
            isinstance(df_copy[col].dtype, pd.CategoricalDtype) or 
            pd.api.types.is_string_dtype(df_copy[col])):
            # Check for strings matching the placeholders (case insensitive, stripped)
            def replace_placeholder(val):
                if pd.isna(val):
                    return np.nan
                s = str(val).strip()
                if s.lower() in placeholders or s == '':
                    return np.nan
                return val
            
            df_copy[col] = df_copy[col].apply(replace_placeholder)
            
    return df_copy

def handle_missing_values(df: pd.DataFrame, column_strategies: dict, custom_values: dict = None) -> tuple[pd.DataFrame, list]:
    """
    Applies imputation or deletion strategies for missing values column-wise.
    column_strategies: dict mapping column_name -> strategy ('remove_rows', 'remove_column', 'mean', 'median', 'mode', 'custom')
    custom_values: dict mapping column_name -> custom value (only used if strategy is 'custom')
    Returns: (cleaned_df, logs_list)
    """
    df_clean = df.copy()
    logs = []
    
    # First, standardize common string null placeholders to np.nan
    df_clean = detect_and_standardize_nulls(df_clean, list(column_strategies.keys()))
    
    columns_to_drop = []
    
    for col, strategy in column_strategies.items():
        if col not in df_clean.columns:
            continue
            
        null_count = df_clean[col].isna().sum()
        if null_count == 0:
            continue
            
        if strategy == 'remove_rows':
            before_len = len(df_clean)
            df_clean = df_clean.dropna(subset=[col])
            after_len = len(df_clean)
            removed = before_len - after_len
            if removed > 0:
                logs.append(f"Removed {removed} rows with missing values in column '{col}'")
                
        elif strategy == 'remove_column':
            columns_to_drop.append(col)
            logs.append(f"Removed column '{col}' because it contained missing values")
            
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                mean_val = df_clean[col].mean()
                if not pd.isna(mean_val):
                    # Cast if necessary to match type
                    df_clean[col] = df_clean[col].fillna(mean_val)
                    logs.append(f"Imputed {null_count} missing values in numeric column '{col}' with Mean value ({mean_val:.2f})")
                else:
                    logs.append(f"Could not compute Mean for column '{col}' (all values null)")
            else:
                logs.append(f"Skipped Mean imputation for column '{col}': Column is not numeric")
                
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                if not pd.isna(median_val):
                    df_clean[col] = df_clean[col].fillna(median_val)
                    logs.append(f"Imputed {null_count} missing values in numeric column '{col}' with Median value ({median_val:.2f})")
                else:
                    logs.append(f"Could not compute Median for column '{col}' (all values null)")
            else:
                logs.append(f"Skipped Median imputation for column '{col}': Column is not numeric")
                
        elif strategy == 'mode':
            if not df_clean[col].dropna().empty:
                mode_val = df_clean[col].mode().iloc[0]
                # Keep categorical types if category is present or convert
                df_clean[col] = df_clean[col].fillna(mode_val)
                logs.append(f"Imputed {null_count} missing values in column '{col}' with Mode value ({str(mode_val)})")
            else:
                logs.append(f"Could not compute Mode for column '{col}' (all values null)")
                
        elif strategy == 'custom':
            custom_val = custom_values.get(col) if custom_values else None
            if custom_val is not None:
                # Convert custom value type to match column type if possible
                try:
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        custom_val = float(custom_val) if '.' in str(custom_val) else int(custom_val)
                except ValueError:
                    pass
                df_clean[col] = df_clean[col].fillna(custom_val)
                logs.append(f"Imputed {null_count} missing values in column '{col}' with custom value: '{custom_val}'")
            else:
                logs.append(f"Skipped custom imputation for column '{col}': No custom value provided")
                
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
        
    return df_clean, logs

def remove_duplicates(df: pd.DataFrame, subset: list = None, keep: str = 'first') -> tuple[pd.DataFrame, list]:
    """
    Identifies and removes duplicate records.
    subset: subset of columns to evaluate. If empty or None, evaluates all columns.
    keep: 'first', 'last', or 'none' (removes all duplicates).
    """
    df_clean = df.copy()
    logs = []
    
    keep_param = False if keep == 'none' else keep
    
    before_len = len(df_clean)
    
    # Calculate duplicate count
    dup_mask = df_clean.duplicated(subset=subset, keep=keep_param)
    dup_count = dup_mask.sum()
    
    if dup_count > 0:
        df_clean = df_clean[~dup_mask]
        after_len = len(df_clean)
        removed = before_len - after_len
        col_desc = "all columns" if not subset else f"columns {subset}"
        logs.append(f"Removed {removed} duplicate records based on {col_desc} (keeping '{keep}')")
    else:
        logs.append("No duplicate records detected based on specified columns")
        
    return df_clean, logs

def parse_phone(val) -> str:
    """
    Helper function to standardize phone numbers to +1-XXX-XXX-XXXX.
    """
    if pd.isna(val):
        return np.nan
        
    s = str(val).strip()
    # Extract only digit characters
    digits = re.sub(r'\D', '', s)
    
    if len(digits) == 10:
        return f"+1-{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
    elif len(digits) == 11 and digits.startswith('1'):
        return f"+1-{digits[1:4]}-{digits[4:7]}-{digits[7:11]}"
    elif len(digits) > 0:
        # If not matching US format but has digits, format as clean string
        if len(digits) > 10:
            return f"+{digits[:-10]}-{digits[-10:-7]}-{digits[-7:-4]}-{digits[-4:]}"
        return digits
    return val

def parse_date(val) -> str:
    """
    Helper function to standardize dates to YYYY-MM-DD.
    """
    if pd.isna(val):
        return np.nan
        
    s = str(val).strip()
    if s == '' or s.lower() in ('nan', 'null', 'none'):
        return np.nan
        
    # Attempt common date parsing via pandas
    try:
        dt = pd.to_datetime(s, errors='coerce')
        if not pd.isna(dt):
            return dt.strftime('%Y-%m-%d')
    except:
        pass
        
    # Regex-based replacements for dot separators or others if pd fails
    try:
        # e.g. 2024.03.12 -> 2024-03-12
        cleaned = re.sub(r'[\.\/]', '-', s)
        dt = pd.to_datetime(cleaned, errors='coerce')
        if not pd.isna(dt):
            return dt.strftime('%Y-%m-%d')
    except:
        pass
        
    return val

def standardize_data(df: pd.DataFrame, operations: dict) -> tuple[pd.DataFrame, list]:
    """
    Standardizes dataframe columns based on operations mapping:
    operations: {
        'names': [col1, col2, ...],
        'dates': [col1, col2, ...],
        'phones': [col1, col2, ...],
        'text_trim': [col1, col2, ...],
        'text_upper': [col1, col2, ...],
        'text_lower': [col1, col2, ...]
    }
    """
    df_clean = df.copy()
    logs = []
    
    if not operations:
        return df_clean, logs
        
    # Names Standardization (Title Case, stripped)
    if 'names' in operations and operations['names']:
        for col in operations['names']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).apply(lambda x: x.strip().title() if pd.notna(x) and x != 'nan' else np.nan)
                logs.append(f"Standardized names capitalization & whitespace in column '{col}'")
                
    # Dates Standardization (YYYY-MM-DD)
    if 'dates' in operations and operations['dates']:
        for col in operations['dates']:
            if col in df_clean.columns:
                # Apply date parser
                df_clean[col] = df_clean[col].apply(parse_date)
                logs.append(f"Standardized date formats to YYYY-MM-DD in column '{col}'")
                
    # Phones Standardization (+1-XXX-XXX-XXXX)
    if 'phones' in operations and operations['phones']:
        for col in operations['phones']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(parse_phone)
                logs.append(f"Standardized phone formats to +1-XXX-XXX-XXXX in column '{col}'")
                
    # Text Trim & extra whitespace
    if 'text_trim' in operations and operations['text_trim']:
        for col in operations['text_trim']:
            if col in df_clean.columns:
                def clean_whitespace(val):
                    if pd.isna(val) or str(val).lower() == 'nan':
                        return np.nan
                    s = str(val).strip()
                    # replace multiple spaces with single space
                    return re.sub(r'\s+', ' ', s)
                df_clean[col] = df_clean[col].apply(clean_whitespace)
                logs.append(f"Trimmed leading/trailing spaces and reduced extra spaces in column '{col}'")
                
    # Casing Operations
    if 'text_upper' in operations and operations['text_upper']:
        for col in operations['text_upper']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).apply(lambda x: x.strip().upper() if pd.notna(x) and x != 'nan' else np.nan)
                logs.append(f"Standardized text to UPPERCASE in column '{col}'")
                
    if 'text_lower' in operations and operations['text_lower']:
        for col in operations['text_lower']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).apply(lambda x: x.strip().lower() if pd.notna(x) and x != 'nan' else np.nan)
                logs.append(f"Standardized text to lowercase in column '{col}'")
                
    return df_clean, logs
