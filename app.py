import streamlit as st
import pandas as pd
import numpy as np
import os
import io

# Import custom modules
from src.profiler import profile_dataframe, infer_column_type
from src.cleaner import (
    handle_missing_values, 
    remove_duplicates, 
    standardize_data, 
    detect_and_standardize_nulls
)
from src.validator import (
    validate_schema, 
    evaluate_custom_rules, 
    detect_outliers_iqr, 
    detect_outliers_zscore, 
    calculate_data_quality_score
)
from src.visualizer import (
    plot_quality_gauge, 
    plot_missing_heatmap, 
    plot_duplicate_pie, 
    plot_datatype_bar, 
    plot_outlier_boxplot, 
    plot_outlier_scatter
)
from src.reporter import (
    generate_pdf_report, 
    df_to_csv_bytes, 
    df_to_excel_bytes, 
    df_to_json_bytes
)

# Page Configuration
st.set_page_config(
    page_title="Intelligent Data Cleaning & Validation",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Modern CSS for Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #1F2937;
    }
    
    /* Sleek card container */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E5E7EB;
        text-align: center;
        margin-bottom: 12px;
    }
    
    .metric-title {
        font-size: 0.8rem;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.5rem;
        color: #111827;
        font-weight: 700;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 2px;
    }
    
    .delta-positive {
        color: #10B981;
    }
    .delta-negative {
        color: #EF4444;
    }
    
    /* Glassmorphism styling for section headers */
    .section-header {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #6366F1;
    }
    
    /* Center align download buttons */
    .stDownloadButton {
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None
    st.session_state['df_raw'] = None
    st.session_state['df_clean'] = None
    st.session_state['cleaning_history'] = []
    st.session_state['schema_config'] = {}
    st.session_state['custom_rules'] = []
    st.session_state['initial_score'] = 100.0
    st.session_state['current_score'] = 100.0
    st.session_state['validation_results'] = None
    st.session_state['validation_ran'] = False
    st.session_state['raw_profile'] = None

# App Header
st.title("🧹 Intelligent Data Cleaning & Structural Validation")
st.markdown("Profile, clean, standardize, and validate your raw datasets dynamically before downstream analysis.")

# File upload in Sidebar
st.sidebar.markdown("### 📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file", 
    type=["csv", "xlsx", "json"],
    help="Supports CSV, Excel (.xlsx), and JSON array datasets."
)

# Helper function to reset application state
def reset_app_state(file_name):
    st.session_state['uploaded_file_name'] = file_name
    st.session_state['df_raw'] = None
    st.session_state['df_clean'] = None
    st.session_state['cleaning_history'] = []
    st.session_state['schema_config'] = {}
    st.session_state['custom_rules'] = []
    st.session_state['initial_score'] = 100.0
    st.session_state['current_score'] = 100.0
    st.session_state['validation_results'] = None
    st.session_state['validation_ran'] = False
    st.session_state['raw_profile'] = None

# Process File Upload
if uploaded_file is not None:
    if st.session_state['uploaded_file_name'] != uploaded_file.name:
        reset_app_state(uploaded_file.name)
        
        # Load Dataset
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        try:
            if file_ext == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext == '.xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_ext == '.json':
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format.")
                df = None
                
            if df is not None:
                # Limit size preview if extremely large, but keep full df
                st.session_state['df_raw'] = df
                st.session_state['df_clean'] = df.copy()
                
                # Run Initial Profile & Inferences
                raw_profile = profile_dataframe(df)
                st.session_state['raw_profile'] = raw_profile
                
                # Auto-initialize schema config based on inferred types
                schema_init = {}
                for col_prof in raw_profile['columns_profile']:
                    schema_init[col_prof['column']] = col_prof['inferred_type']
                st.session_state['schema_config'] = schema_init
                
                # Calculate initial quality scores
                init_score_dict = calculate_data_quality_score(df, schema_init)
                st.session_state['initial_score'] = init_score_dict['overall_score']
                st.session_state['current_score'] = init_score_dict['overall_score']
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

# Get current dataframes
df_raw = st.session_state['df_raw']
df_clean = st.session_state['df_clean']

# Sidebar Metrics Section (Only shown when data is loaded)
if df_raw is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Status")
    
    # Render sidebars quality delta card
    init_score = st.session_state['initial_score']
    curr_score = st.session_state['current_score']
    score_delta = curr_score - init_score
    
    delta_class = "delta-positive" if score_delta >= 0 else "delta-negative"
    delta_sign = "+" if score_delta >= 0 else ""
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Data Quality Score</div>
        <div class="metric-value">{curr_score:.1f}%</div>
        <div class="metric-delta {delta_class}">{delta_sign}{score_delta:.1f}% since upload</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Rows & Columns
    raw_rows, raw_cols = df_raw.shape
    clean_rows, clean_cols = df_clean.shape
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Dataset Dimensions</div>
        <div class="metric-value">{clean_rows:,} x {clean_cols}</div>
        <div class="metric-title" style="margin-top:8px; font-size:0.7rem;">Original: {raw_rows:,} x {raw_cols}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Validation status
    val_ran = st.session_state['validation_ran']
    val_results = st.session_state['validation_results']
    
    if val_ran and val_results:
        schema_valid = val_results.get("schema_valid", True)
        rules_valid = val_results.get("rules_valid", True)
        
        status_text = "PASSED" if (schema_valid and rules_valid) else "FAILED"
        status_color = "#10B981" if (schema_valid and rules_valid) else "#EF4444"
        
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Validation Status</div>
            <div class="metric-value" style="color: {status_color};">{status_text}</div>
            <div class="metric-title" style="margin-top:8px; font-size:0.7rem;">Schema: {"✓" if schema_valid else "✗"} | Rules: {"✓" if rules_valid else "✗"}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="metric-card">
            <div class="metric-title">Validation Status</div>
            <div class="metric-value" style="color: #6B7280; font-size:1.2rem;">Not Checked</div>
        </div>
        """, unsafe_allow_html=True)

# Main Dashboard Navigation Tabs
tab_upload, tab_profile, tab_cleaning, tab_validation, tab_dashboard, tab_reports = st.tabs([
    "📂 Dataset Upload", 
    "🔍 Data Profiling", 
    "🧹 Data Cleaning", 
    "✅ Structural Validation", 
    "📊 Quality Dashboard", 
    "📄 Reports"
])

# ================= TAB 1: DATASET UPLOAD =================
with tab_upload:
    if df_raw is None:
        st.info("Please upload a CSV, Excel, or JSON dataset in the sidebar to get started.")
        # Visual Mockup/Hero Section for Empty State
        st.markdown("### Platform Capabilities")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.subheader("🔍 Real-time Profiling")
            st.write("Extract column summaries, missing value counts, and logical data types automatically.")
        with col_c2:
            st.subheader("🧹 Intelligent Cleaning")
            st.write("Impute nulls, remove duplicate records, and standardize casing, phone numbers, and dates.")
        with col_c3:
            st.subheader("✅ Schema & Rule Validation")
            st.write("Define expected schemas and configure custom logical rules to catch anomalous rows.")
    else:
        st.markdown("<div class='section-header'><h3>Dataset Preview & Metadata</h3></div>", unsafe_allow_html=True)
        
        # Display file name and details
        file_size_mb = uploaded_file.size / (1024 * 1024)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("File Name", uploaded_file.name)
        col_m2.metric("File Size", f"{file_size_mb:.2f} MB")
        col_m3.metric("Original Rows", f"{raw_rows:,}")
        col_m4.metric("Original Columns", f"{raw_cols}")
        
        st.markdown("#### Sample Preview (Top 100 rows)")
        st.dataframe(df_clean.head(100), use_container_width=True)

# ================= TAB 2: DATA PROFILING =================
with tab_profile:
    if df_raw is None:
        st.info("Upload a dataset to run profiling analyses.")
    else:
        st.markdown("<div class='section-header'><h3>Dataset Profiling & Logical Types</h3></div>", unsafe_allow_html=True)
        
        # Re-profile on clean df to capture changes
        profile_res = profile_dataframe(df_clean)
        
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Overall Completeness", f"{profile_res['overall_completeness']:.1f}%")
        col_p2.metric("Duplicate Rows", f"{profile_res['duplicate_rows']:,}")
        col_p3.metric("Duplicate Percentage", f"{profile_res['duplicate_percentage']:.2f}%")
        
        # Details of columns
        st.markdown("#### Column Metadata & Type Diagnostics")
        col_details = []
        for col_p in profile_res['columns_profile']:
            # Stats representation
            stats = col_p['stats']
            if stats:
                stats_str = f"Mean: {stats['mean']:.2f} | Med: {stats['median']:.2f} | Range: [{stats['min']:.1f}, {stats['max']:.1f}]"
            else:
                stats_str = "N/A"
                
            # Sample values
            top_vals = [f"'{item['value']}': {item['count']}" for item in col_p['top_values'][:3]]
            sample_str = ", ".join(top_vals)
            
            col_details.append({
                "Column Name": col_p['column'],
                "Pandas Type": col_p['type'],
                "Inferred Logical Type": col_p['inferred_type'],
                "Missing Cells": col_p['missing_count'],
                "Missing %": f"{col_p['missing_percentage']:.2f}%",
                "Unique Values": col_p['unique_count'],
                "Summary Statistics": stats_str,
                "Top Frequency Samples": sample_str
            })
            
        st.dataframe(pd.DataFrame(col_details), use_container_width=True, hide_index=True)
        
        # Draw Data Type Distribution
        inferred_types = {c['column']: c['inferred_type'] for c in profile_res['columns_profile']}
        fig_dtypes = plot_datatype_bar(inferred_types)
        st.plotly_chart(fig_dtypes, use_container_width=True)

# ================= TAB 3: DATA CLEANING =================
with tab_cleaning:
    if df_raw is None:
        st.info("Upload a dataset to perform cleaning operations.")
    else:
        st.markdown("<div class='section-header'><h3>🧹 Data Cleaning Operations</h3></div>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("#### 1. Handle Missing Values")
            # Build list of missing columns
            profile_res = profile_dataframe(df_clean)
            miss_cols = [c['column'] for c in profile_res['columns_profile'] if c['missing_count'] > 0]
            
            if not miss_cols:
                st.success("No missing values detected in the current dataset.")
            else:
                select_miss_col = st.selectbox("Select Column to Impute", miss_cols)
                impute_strategy = st.selectbox(
                    "Imputation Strategy",
                    ["mean", "median", "mode", "remove_rows", "remove_column", "custom"]
                )
                
                custom_val = None
                if impute_strategy == "custom":
                    custom_val = st.text_input("Enter Custom Imputation Value")
                    
                if st.button("Apply Imputation", use_container_width=True):
                    strategy_map = {select_miss_col: impute_strategy}
                    custom_map = {select_miss_col: custom_val} if custom_val else {}
                    
                    df_clean, logs = handle_missing_values(df_clean, strategy_map, custom_map)
                    st.session_state['df_clean'] = df_clean
                    st.session_state['cleaning_history'].extend(logs)
                    
                    # Update Quality Score
                    score_res = calculate_data_quality_score(df_clean, st.session_state['schema_config'], st.session_state['custom_rules'])
                    st.session_state['current_score'] = score_res['overall_score']
                    st.success("Imputation complete!")
                    st.rerun()
                    
            st.markdown("---")
            st.markdown("#### 2. Deduplicate Records")
            dup_cols_choice = st.multiselect(
                "Deduplicate subset (Leave empty for all columns)",
                list(df_clean.columns),
                help="If no columns selected, duplicates will be detected across all columns."
            )
            keep_choice = st.selectbox("Record to keep", ["first", "last", "none"])
            
            if st.button("Run Deduplication", use_container_width=True):
                subset = dup_cols_choice if len(dup_cols_choice) > 0 else None
                df_clean, logs = remove_duplicates(df_clean, subset=subset, keep=keep_choice)
                st.session_state['df_clean'] = df_clean
                st.session_state['cleaning_history'].extend(logs)
                
                # Update Quality Score
                score_res = calculate_data_quality_score(df_clean, st.session_state['schema_config'], st.session_state['custom_rules'])
                st.session_state['current_score'] = score_res['overall_score']
                st.success("Deduplication complete!")
                st.rerun()
                
            st.markdown("---")
            st.markdown("#### 3. Standardize Formatting")
            standardize_type = st.selectbox(
                "Select Format Style",
                ["Capitalize Names (John Doe)", "Date Format (YYYY-MM-DD)", "Phone Numbers (+1-XXX-XXX-XXXX)", "Text Trim & Clean Whitespace", "Text UPPERCASE", "Text lowercase"]
            )
            
            target_cols = st.multiselect("Select columns to standardize", list(df_clean.columns))
            
            if st.button("Apply Standardization", use_container_width=True):
                if not target_cols:
                    st.warning("Please select at least one column.")
                else:
                    ops = {}
                    if standardize_type == "Capitalize Names (John Doe)":
                        ops['names'] = target_cols
                    elif standardize_type == "Date Format (YYYY-MM-DD)":
                        ops['dates'] = target_cols
                    elif standardize_type == "Phone Numbers (+1-XXX-XXX-XXXX)":
                        ops['phones'] = target_cols
                    elif standardize_type == "Text Trim & Clean Whitespace":
                        ops['text_trim'] = target_cols
                    elif standardize_type == "Text UPPERCASE":
                        ops['text_upper'] = target_cols
                    elif standardize_type == "Text lowercase":
                        ops['text_lower'] = target_cols
                        
                    df_clean, logs = standardize_data(df_clean, ops)
                    st.session_state['df_clean'] = df_clean
                    st.session_state['cleaning_history'].extend(logs)
                    
                    # Update Quality Score
                    score_res = calculate_data_quality_score(df_clean, st.session_state['schema_config'], st.session_state['custom_rules'])
                    st.session_state['current_score'] = score_res['overall_score']
                    st.success("Standardization complete!")
                    st.rerun()
                    
            st.markdown("---")
            if st.button("Reset Cleaned Dataset", type="secondary", use_container_width=True):
                st.session_state['df_clean'] = df_raw.copy()
                st.session_state['cleaning_history'] = []
                st.session_state['current_score'] = st.session_state['initial_score']
                st.session_state['validation_ran'] = False
                st.success("Dataset reset to original state!")
                st.rerun()

        with col_right:
            st.markdown("#### Active Cleaning Logs & History")
            if len(st.session_state['cleaning_history']) == 0:
                st.info("No cleaning operations applied yet. Use the left panel to execute operations.")
            else:
                for idx, log in enumerate(st.session_state['cleaning_history'], 1):
                    st.markdown(f"**{idx}.** {log}")
                    
            st.markdown("---")
            st.markdown("#### Cleaned Dataset Preview")
            st.dataframe(df_clean.head(100), use_container_width=True)

# ================= TAB 4: STRUCTURAL VALIDATION =================
with tab_validation:
    if df_raw is None:
        st.info("Upload a dataset to validate schemas and rules.")
    else:
        st.markdown("<div class='section-header'><h3>✅ Structural Schema & Custom Rules Engine</h3></div>", unsafe_allow_html=True)
        
        col_v_left, col_v_right = st.columns([1, 2])
        
        with col_v_left:
            st.markdown("#### 1. Define Expected Column Types")
            current_schema = st.session_state['schema_config']
            
            schema_updates = {}
            for col in df_clean.columns:
                prev_type = current_schema.get(col, "String")
                schema_updates[col] = st.selectbox(
                    f"Column '{col}' expected type:",
                    ["Integer", "Float", "String", "Date", "Boolean"],
                    index=["Integer", "Float", "String", "Date", "Boolean"].index(prev_type),
                    key=f"schema_sel_{col}"
                )
                
            st.session_state['schema_config'] = schema_updates
            
            st.markdown("---")
            st.markdown("#### 2. Create Logical Rules")
            rule_col = st.selectbox("Select Rule Column", list(df_clean.columns))
            rule_op = st.selectbox(
                "Select Operator",
                [">=", ">", "<=", "<", "==", "!=", "contains", "length ==", "is_email", "is_not_null"]
            )
            rule_val = st.text_input("Comparison Value (if applicable)")
            
            if st.button("Add Rule", use_container_width=True):
                new_rule = {"column": rule_col, "operator": rule_op, "value": rule_val}
                st.session_state['custom_rules'].append(new_rule)
                st.success("Rule added successfully!")
                
            # List current rules
            if st.session_state['custom_rules']:
                st.markdown("**Configured Rules:**")
                for r_idx, r in enumerate(st.session_state['custom_rules']):
                    val_str = f" '{r['value']}'" if r['operator'] not in ('is_email', 'is_not_null') else ""
                    st.markdown(f"- `{r['column']}` {r['operator']}{val_str}")
                if st.button("Clear Rules", use_container_width=True):
                    st.session_state['custom_rules'] = []
                    st.success("All custom rules cleared.")
                    st.rerun()
                    
            st.markdown("---")
            if st.button("Run Complete Validation", type="primary", use_container_width=True):
                schema_res = validate_schema(df_clean, st.session_state['schema_config'])
                rules_res = evaluate_custom_rules(df_clean, st.session_state['custom_rules'])
                
                st.session_state['validation_results'] = {
                    "schema_valid": schema_res["is_valid"],
                    "schema_errors": schema_res["errors"],
                    "missing_cols": schema_res["missing_columns"],
                    "unexpected_cols": schema_res["unexpected_columns"],
                    "rules_valid": rules_res["is_valid"],
                    "rules_results": rules_res["results"],
                    "rules_failed_rows": rules_res["failed_rows"]
                }
                st.session_state['validation_ran'] = True
                
                # Re-calculate Data Quality Score
                score_res = calculate_data_quality_score(df_clean, st.session_state['schema_config'], st.session_state['custom_rules'])
                st.session_state['current_score'] = score_res['overall_score']
                st.success("Validation Completed!")
                st.rerun()

        with col_v_right:
            st.markdown("#### Outlier Detection")
            num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            
            if not num_cols:
                st.info("No numerical columns available for outlier detection.")
            else:
                col_o_c1, col_o_c2, col_o_c3 = st.columns(3)
                outlier_col = col_o_c1.selectbox("Outlier Target Column", num_cols)
                outlier_method = col_o_c2.selectbox("Outlier Method", ["IQR Method", "Z-Score Method"])
                outlier_thresh = col_o_c3.number_input(
                    "Threshold multiplier", 
                    value=1.5 if outlier_method == "IQR Method" else 3.0,
                    step=0.1
                )
                
                if outlier_method == "IQR Method":
                    mask, summary = detect_outliers_iqr(df_clean, outlier_col)
                    st.write(f"IQR Bounds: Lower={summary['lower_bound']:.2f}, Upper={summary['upper_bound']:.2f} | Outliers Count: **{summary['count']}**")
                else:
                    mask, summary = detect_outliers_zscore(df_clean, outlier_col, outlier_thresh)
                    st.write(f"Z-Score Mean: {summary['mean']:.2f}, Std: {summary['std']:.2f} | Outliers Count: **{summary['count']}**")
                    
                if summary['count'] > 0:
                    fig_box = plot_outlier_boxplot(df_clean, outlier_col)
                    fig_scatter = plot_outlier_scatter(df_clean, outlier_col, mask)
                    
                    col_chart1, col_chart2 = st.columns(2)
                    col_chart1.plotly_chart(fig_box, use_container_width=True)
                    col_chart2.plotly_chart(fig_scatter, use_container_width=True)
                    
                    st.markdown("**Sample Outlier Records:**")
                    st.dataframe(df_clean[mask].head(50), use_container_width=True)
                else:
                    st.success("No outliers detected in the target column.")
                    
            st.markdown("---")
            st.markdown("#### Validation Output Results")
            if not st.session_state['validation_ran']:
                st.info("Click 'Run Complete Validation' in the left panel to view structural verification results.")
            else:
                res = st.session_state['validation_results']
                
                # Schema errors
                st.markdown("##### 📁 Schema Consistency Results")
                if res["schema_valid"]:
                    st.success("✓ All columns match their expected schemas!")
                else:
                    st.error("✗ Schema violations or missing columns detected!")
                    if res["missing_cols"]:
                        st.markdown(f"**Missing Columns:** {', '.join(res['missing_cols'])}")
                    if res["unexpected_cols"]:
                        st.markdown(f"**Unexpected Columns:** {', '.join(res['unexpected_cols'])}")
                    if res["schema_errors"]:
                        st.dataframe(pd.DataFrame(res["schema_errors"]), use_container_width=True)
                        
                # Custom rules errors
                st.markdown("##### ⚙ Custom Rules Evaluation")
                if not st.session_state['custom_rules']:
                    st.info("No custom rules configured yet.")
                elif res["rules_valid"]:
                    st.success("✓ All custom rules evaluated successfully with 0 failures!")
                else:
                    st.error("✗ Some custom rules failed validation!")
                    st.dataframe(pd.DataFrame(res["rules_results"]), use_container_width=True)
                    if res["rules_failed_rows"]:
                        st.markdown("**Sample Failing Records:**")
                        st.dataframe(pd.DataFrame(res["rules_failed_rows"]), use_container_width=True)

# ================= TAB 5: QUALITY DASHBOARD =================
with tab_dashboard:
    if df_raw is None:
        st.info("Upload a dataset to view the quality metrics visual dashboard.")
    else:
        st.markdown("<div class='section-header'><h3>📊 Data Quality Visual Dashboard</h3></div>", unsafe_allow_html=True)
        
        # Calculate full quality score breakdown
        q_breakdown = calculate_data_quality_score(
            df_clean, 
            st.session_state['schema_config'], 
            st.session_state['custom_rules']
        )
        
        col_d1, col_d2 = st.columns([1, 2])
        
        with col_d1:
            fig_gauge = plot_quality_gauge(q_breakdown["overall_score"])
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Show four breakdown dimensions
            st.markdown("#### Quality Breakdown Dimensions")
            st.markdown(f"- **Completeness (no null cells):** {q_breakdown['completeness']}%")
            st.markdown(f"- **Uniqueness (no duplicate rows):** {q_breakdown['uniqueness']}%")
            st.markdown(f"- **Validity (schema matching):** {q_breakdown['validity']}%")
            st.markdown(f"- **Consistency (custom rules passing):** {q_breakdown['consistency']}%")
            
        with col_d2:
            fig_missing = plot_missing_heatmap(df_clean)
            st.plotly_chart(fig_missing, use_container_width=True)
            
        col_d3, col_d4 = st.columns(2)
        with col_d3:
            fig_pie = plot_duplicate_pie(df_clean)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_d4:
            # Re-fetch profiling details
            profile_res = profile_dataframe(df_clean)
            inferred_types = {c['column']: c['inferred_type'] for c in profile_res['columns_profile']}
            fig_types = plot_datatype_bar(inferred_types)
            st.plotly_chart(fig_types, use_container_width=True)

# ================= TAB 6: REPORTS & EXPORTS =================
with tab_reports:
    if df_raw is None:
        st.info("Upload a dataset to generate reports and download cleaned files.")
    else:
        st.markdown("<div class='section-header'><h3>📄 Download Cleaned Data & Automated Reports</h3></div>", unsafe_allow_html=True)
        
        col_r_left, col_r_right = st.columns(2)
        
        with col_r_left:
            st.markdown("#### Cleaned Dataset Exporter")
            st.write("Export your processed dataset in CSV, Excel, or JSON format. All imputations, deduplications, and formatting changes are preserved.")
            
            # File name suggestions
            base, ext = os.path.splitext(uploaded_file.name)
            cleaned_filename = f"{base}_cleaned"
            
            # Export CSV button
            csv_bytes = df_to_csv_bytes(df_clean)
            st.download_button(
                label="📥 Download CSV Dataset",
                data=csv_bytes,
                file_name=f"{cleaned_filename}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Export Excel button
            excel_bytes = df_to_excel_bytes(df_clean)
            st.download_button(
                label="📥 Download Excel Dataset",
                data=excel_bytes,
                file_name=f"{cleaned_filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Export JSON button
            json_bytes = df_to_json_bytes(df_clean)
            st.download_button(
                label="📥 Download JSON Dataset",
                data=json_bytes,
                file_name=f"{cleaned_filename}.json",
                mime="application/json",
                use_container_width=True
            )

        with col_r_right:
            st.markdown("#### Automated Quality Report")
            st.write("Download a professional PDF quality report containing before-and-after scores, rows/columns comparisons, validation error logs, and cleaning history.")
            
            # Compile stats
            dup_raw = df_raw.duplicated().sum()
            dup_clean = df_clean.duplicated().sum()
            
            summary_stats = {
                "initial_rows": raw_rows,
                "final_rows": clean_rows,
                "initial_cols": raw_cols,
                "final_cols": clean_cols,
                "initial_duplicates": dup_raw,
                "final_duplicates": dup_clean
            }
            
            validation_summary = {}
            if st.session_state['validation_ran'] and st.session_state['validation_results']:
                res = st.session_state['validation_results']
                
                # Count total rule failures across all custom rules
                failed_rules = sum([1 for r in res.get("rules_results", []) if r["status"] == "Failed"])
                
                validation_summary = {
                    "schema_valid": res["schema_valid"],
                    "total_rules": len(st.session_state['custom_rules']),
                    "failed_rules": failed_rules,
                    "errors": res.get("schema_errors", []) + res.get("rules_failed_rows", [])
                }
                
            # PDF Generation
            try:
                pdf_report = generate_pdf_report(
                    summary_stats=summary_stats,
                    cleaning_logs=st.session_state['cleaning_history'],
                    validation_summary=validation_summary,
                    initial_score=st.session_state['initial_score'],
                    final_score=st.session_state['current_score']
                )
                
                st.download_button(
                    label="📥 Download Automated PDF Quality Report",
                    data=pdf_report,
                    file_name=f"Data_Cleaning_Report_{base}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
            except Exception as e:
                st.error(f"Error compiling PDF report: {str(e)}")
                
            st.markdown("---")
            st.markdown("##### Processing Summary Check")
            st.write(f"- Operations applied: **{len(st.session_state['cleaning_history'])}**")
            st.write(f"- Data quality improved from **{st.session_state['initial_score']:.1f}%** to **{st.session_state['current_score']:.1f}%**")
