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
    
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #1F2937;
        font-size: 2.2rem !important;
        line-height: 1.2;
    }
    
    h2, h3, h4, h5, h6 {
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
    
    /* Premium Quality Card */
    .quality-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        border: 1px solid #E5E7EB;
        display: flex;
        flex-direction: row;
        gap: 32px;
        margin-top: 10px;
        margin-bottom: 24px;
        align-items: center;
    }
    
    .quality-left {
        flex: 1;
        text-align: center;
        border-right: 1px solid #E5E7EB;
        padding-right: 32px;
    }
    
    .quality-right {
        flex: 2;
    }
    
    .overall-title {
        font-size: 0.85rem;
        color: #6B7280;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .overall-score {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 8px;
        font-family: 'Outfit', sans-serif;
    }
    
    .score-out-of {
        font-size: 1.5rem;
        color: #9CA3AF;
        font-weight: 500;
    }
    
    .overall-status {
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-excellent { color: #10B981; }
    .status-good { color: #F59E0B; }
    .status-poor { color: #EF4444; }
    
    .breakdown-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #374151;
        margin-bottom: 16px;
        font-family: 'Outfit', sans-serif;
    }
    
    .breakdown-item {
        margin-bottom: 12px;
    }
    
    .breakdown-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        color: #4B5563;
        margin-bottom: 4px;
    }
    
    .progress-bar {
        background-color: #E5E7EB;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s ease-in-out;
    }
    
    @media (max-width: 768px) {
        .quality-card {
            flex-direction: column;
            gap: 20px;
        }
        .quality-left {
            border-right: none;
            padding-right: 0;
            border-bottom: 1px solid #E5E7EB;
            padding-bottom: 20px;
            width: 100%;
        }
        .quality-right {
            width: 100%;
        }
    }
    
    /* KPI Card styling */
    .kpi-row {
        display: flex;
        flex-direction: row;
        gap: 16px;
        margin-top: 15px;
        margin-bottom: 20px;
        width: 100%;
    }
    
    .kpi-card {
        flex: 1;
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    }
    
    .kpi-title {
        font-size: 0.8rem;
        color: #6B7280;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        color: #111827;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
    }
    
    @media (max-width: 768px) {
        .kpi-row {
            flex-direction: column;
            gap: 12px;
        }
    }
    
    /* Before vs After Table */
    .comparison-table-container {
        margin-top: 15px;
        margin-bottom: 24px;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        background-color: #ffffff;
    }
    
    .comparison-table th {
        background-color: #374151;
        color: #ffffff;
        padding: 12px 16px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .comparison-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #E5E7EB;
        color: #4B5563;
        font-size: 0.95rem;
    }
    
    .comparison-table tr:last-child td {
        border-bottom: none;
    }
    
    .comparison-table tr:hover {
        background-color: #F9FAFB;
    }
    
    .comparison-metric {
        font-weight: 600;
        color: #111827 !important;
    }
    
    .val-before {
        color: #EF4444 !important;
        font-weight: 600;
    }
    
    .val-after {
        color: #10B981 !important;
        font-weight: 700;
    }
    
    .val-neutral {
        color: #4B5563 !important;
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

def render_comparison_table(df_raw, df_clean, initial_score, current_score, schema_config, custom_rules):
    """
    Renders a before vs after comparison table of data quality metrics.
    """
    # 1. Missing Values
    raw_profile = profile_dataframe(df_raw)
    raw_missing = sum(c['missing_count'] for c in raw_profile['columns_profile'])
    clean_profile = profile_dataframe(df_clean)
    clean_missing = sum(c['missing_count'] for c in clean_profile['columns_profile'])
    
    # 2. Duplicates
    raw_dupes = df_raw.duplicated().sum()
    clean_dupes = df_clean.duplicated().sum()
    
    # 3. Invalid Values (Type mismatches + custom rules)
    raw_val = validate_schema(df_raw, schema_config)
    raw_invalid = len(raw_val["errors"])
    if custom_rules:
        raw_r = evaluate_custom_rules(df_raw, custom_rules)
        raw_invalid += len(raw_r["failed_rows"])
        
    clean_val = validate_schema(df_clean, schema_config)
    clean_invalid = len(clean_val["errors"])
    if custom_rules:
        clean_r = evaluate_custom_rules(df_clean, custom_rules)
        clean_invalid += len(clean_r["failed_rows"])
        
    # Styling classes for text color indicators
    missing_after_class = "val-after" if clean_missing == 0 else ("val-before" if clean_missing > raw_missing else "val-neutral")
    dupes_after_class = "val-after" if clean_dupes == 0 else ("val-before" if clean_dupes > raw_dupes else "val-neutral")
    invalid_after_class = "val-after" if clean_invalid == 0 else ("val-before" if clean_invalid > raw_invalid else "val-neutral")
    score_after_class = "val-after" if current_score >= initial_score else "val-before"
    
    st.markdown(f"""
    <div class="comparison-table-container">
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Before Cleaning</th>
                    <th>After Cleaning</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="comparison-metric">Missing Values</td>
                    <td class="val-before">{raw_missing:,}</td>
                    <td class="{missing_after_class}">{clean_missing:,}</td>
                </tr>
                <tr>
                    <td class="comparison-metric">Duplicate Records</td>
                    <td class="val-before">{raw_dupes:,}</td>
                    <td class="{dupes_after_class}">{clean_dupes:,}</td>
                </tr>
                <tr>
                    <td class="comparison-metric">Invalid Values</td>
                    <td class="val-before">{raw_invalid:,}</td>
                    <td class="{invalid_after_class}">{clean_invalid:,}</td>
                </tr>
                <tr>
                    <td class="comparison-metric">Data Quality Score</td>
                    <td class="val-before">{initial_score:.1f}%</td>
                    <td class="{score_after_class}">{current_score:.1f}%</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

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
        
        # Calculate clean profile details for KPI row
        clean_profile = profile_dataframe(df_clean)
        clean_dupes = df_clean.duplicated().sum()
        clean_missing = sum(col_prof["missing_count"] for col_prof in clean_profile["columns_profile"])
        
        st.markdown(f"📁 **Uploaded File:** `{uploaded_file.name}` ({file_size_mb:.2f} MB)")
        
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">Rows</div>
                <div class="kpi-value">{clean_rows:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Columns</div>
                <div class="kpi-value">{clean_cols}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Missing Cells</div>
                <div class="kpi-value">{clean_missing:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Duplicate Rows</div>
                <div class="kpi-value">{clean_dupes:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate quality score breakdown
        q_breakdown = calculate_data_quality_score(
            df_clean, 
            st.session_state['schema_config'], 
            st.session_state['custom_rules']
        )
        score = q_breakdown["overall_score"]
        
        # Determine status text and class
        if score >= 90:
            status_text = "Excellent"
            status_class = "status-excellent"
        elif score >= 70:
            status_text = "Good"
            status_class = "status-good"
        else:
            status_text = "Poor"
            status_class = "status-poor"
            
        st.markdown(f"""
        <div class="quality-card">
            <div class="quality-left">
                <div class="overall-title">Data Quality Score</div>
                <div class="overall-score {status_class}">{score:.0f}<span class="score-out-of">/100</span></div>
                <div class="overall-status {status_class}">{status_text}</div>
            </div>
            <div class="quality-right">
                <div class="breakdown-title">Quality Breakdown</div>
                <div class="breakdown-item">
                    <div class="breakdown-label">
                        <span>Completeness</span>
                        <span>{q_breakdown['completeness']:.0f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {q_breakdown['completeness']}%; background-color: #10B981;"></div>
                    </div>
                </div>
                <div class="breakdown-item">
                    <div class="breakdown-label">
                        <span>Consistency</span>
                        <span>{q_breakdown['consistency']:.0f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {q_breakdown['consistency']}%; background-color: #3B82F6;"></div>
                    </div>
                </div>
                <div class="breakdown-item">
                    <div class="breakdown-label">
                        <span>Validity</span>
                        <span>{q_breakdown['validity']:.0f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {q_breakdown['validity']}%; background-color: #8B5CF6;"></div>
                    </div>
                </div>
                <div class="breakdown-item">
                    <div class="breakdown-label">
                        <span>Uniqueness</span>
                        <span>{q_breakdown['uniqueness']:.0f}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {q_breakdown['uniqueness']}%; background-color: #EC4899;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📥 Raw Dataset Preview (First 10 rows)")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        st.markdown("#### 🧹 Current Cleaned Dataset Preview (Top 100 rows)")
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
                
                # Schema Validation Results
                st.markdown("##### 📁 Schema Consistency Results")
                schema_config = st.session_state['schema_config']
                schema_errors = res.get("schema_errors", [])
                
                # Render check items for each column in schema config
                for col_name, expected_type in schema_config.items():
                    if col_name in res.get("missing_cols", []):
                        st.error(f"✗ Column '{col_name}' is missing from the dataset")
                    else:
                        # Count errors for this column
                        col_errors = [err for err in schema_errors if err["column"] == col_name]
                        err_count = len(col_errors)
                        if err_count == 0:
                            st.success(f"✓ {col_name} datatype valid (conforms to {expected_type})")
                        else:
                            st.error(f"✗ {err_count} rows contain invalid {col_name} datatype (expected {expected_type})")
                
                if res.get("unexpected_cols", []):
                    st.warning(f"⚠️ Unexpected Columns present: {', '.join(res['unexpected_cols'])}")
                
                if schema_errors:
                    with st.expander("Show detailed schema error logs"):
                        st.dataframe(pd.DataFrame(schema_errors), use_container_width=True)
                        
                # Custom Rules Results
                st.markdown("##### ⚙ Custom Rules Evaluation")
                if not st.session_state['custom_rules']:
                    st.info("No custom rules configured yet.")
                else:
                    rules_results = res.get("rules_results", [])
                    rules_failed_rows = res.get("rules_failed_rows", [])
                    
                    for r_res in rules_results:
                        rule_str = r_res["rule"]
                        status = r_res["status"]
                        failures = r_res.get("failures", 0)
                        msg = r_res.get("message", "")
                        
                        if status == "Passed":
                            st.success(f"✓ Rule {rule_str} valid (0 failures)")
                        elif status == "Failed":
                            st.error(f"✗ {failures} rows failed rule: {rule_str}")
                        else:
                            st.warning(f"⚠️ Rule {rule_str} error: {msg}")
                            
                    if rules_failed_rows:
                        with st.expander("Show detailed custom rules failure logs"):
                            st.dataframe(pd.DataFrame(rules_failed_rows), use_container_width=True)

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
            
            st.markdown("#### Before vs After Comparison")
            render_comparison_table(
                df_raw, 
                df_clean, 
                st.session_state['initial_score'], 
                st.session_state['current_score'],
                st.session_state['schema_config'],
                st.session_state['custom_rules']
            )
            
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
            
        # Outlier Analysis Section
        st.markdown("---")
        st.markdown("#### 📈 Outlier Analysis Dashboard")
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numerical columns available for outlier detection.")
        else:
            col_o1, col_o2 = st.columns([1, 3])
            with col_o1:
                outlier_target = st.selectbox(
                    "Select Column for Outlier Analysis", 
                    num_cols, 
                    key="dashboard_outlier_col"
                )
                method = st.selectbox(
                    "Outlier Detection Method", 
                    ["IQR Method", "Z-Score Method"], 
                    key="dashboard_outlier_method"
                )
                thresh = st.number_input(
                    "Threshold multiplier", 
                    value=1.5 if method == "IQR Method" else 3.0, 
                    step=0.1, 
                    key="dashboard_outlier_thresh"
                )
                
                # Compute outlier count
                if method == "IQR Method":
                    mask, summary = detect_outliers_iqr(df_clean, outlier_target)
                else:
                    mask, summary = detect_outliers_zscore(df_clean, outlier_target, thresh)
                st.metric("Detected Outliers", f"{summary['count']}")
                
            with col_o2:
                fig_box = plot_outlier_boxplot(df_clean, outlier_target)
                fig_scatter = plot_outlier_scatter(df_clean, outlier_target, mask)
                
                col_chart1, col_chart2 = st.columns(2)
                col_chart1.plotly_chart(fig_box, use_container_width=True)
                col_chart2.plotly_chart(fig_scatter, use_container_width=True)

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
            st.markdown("##### Before vs After Cleaning Comparison")
            render_comparison_table(
                df_raw, 
                df_clean, 
                st.session_state['initial_score'], 
                st.session_state['current_score'],
                st.session_state['schema_config'],
                st.session_state['custom_rules']
            )
            st.write(f"Operations applied: **{len(st.session_state['cleaning_history'])}**")
