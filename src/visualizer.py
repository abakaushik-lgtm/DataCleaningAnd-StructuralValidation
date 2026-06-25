import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from src.cleaner import detect_and_standardize_nulls

def plot_quality_gauge(score: float) -> go.Figure:
    """
    Renders a Plotly gauge chart representing the Data Quality Score.
    """
    # Sleek color scheme based on score
    if score >= 90:
        gauge_color = "#10B981"  # Emerald Green
    elif score >= 70:
        gauge_color = "#F59E0B"  # Amber/Yellow
    else:
        gauge_color = "#EF4444"  # Red
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Data Quality Score", 'font': {'size': 20, 'family': 'Outfit, Inter, Arial'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4B5563"},
            'bar': {'color': gauge_color},
            'bgcolor': "#F3F4F6",
            'borderwidth': 2,
            'bordercolor': "#E5E7EB",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1F2937", 'family': "Outfit, Inter"},
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Renders an interactive missing values heatmap using Plotly.
    Rows are sampled if the dataset is very large to maintain high performance.
    """
    if df is None or df.empty:
        return go.Figure()
        
    # Standardize nulls for visualization
    df_nulls = detect_and_standardize_nulls(df)
    
    # Calculate null mask (1 for missing, 0 for present)
    null_mask = df_nulls.isna().astype(int)
    
    # Sample rows if dataset is large (> 2000 rows) for smooth plotting
    if len(null_mask) > 2000:
        null_mask_sample = null_mask.sample(n=2000, random_state=42).sort_index()
        title_suffix = " (Sampled 2,000 Rows)"
    else:
        null_mask_sample = null_mask
        title_suffix = ""
        
    # Heatmap color scale: Present = Deep slate, Missing = Vibrant Red/Coral
    colorscale = [[0.0, '#374151'], [1.0, '#EF4444']]
    
    fig = go.Figure(data=go.Heatmap(
        z=null_mask_sample.values,
        x=null_mask_sample.columns,
        y=null_mask_sample.index,
        colorscale=colorscale,
        showscale=False,
        xgap=1,
        ygap=0,
        hovertemplate="Column: %{x}<br>Row: %{y}<br>Status: %{customdata}<extra></extra>",
        customdata=np.where(null_mask_sample.values == 1, "Missing (NaN)", "Present")
    ))
    
    fig.update_layout(
        title=f"Missing Values Heatmap{title_suffix}<br><sup><span style='color:#EF4444'>Red: Missing</span> | <span style='color:#374151'>Grey: Present</span></sup>",
        xaxis_title="Columns",
        yaxis_title="Row Index",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1F2937", 'family': "Outfit, Inter"},
        margin=dict(l=20, r=20, t=60, b=20),
        height=320
    )
    
    return fig

def plot_duplicate_pie(df: pd.DataFrame) -> go.Figure:
    """
    Renders a pie chart showing proportion of unique vs duplicate records.
    """
    if df is None or df.empty:
        return go.Figure()
        
    total = len(df)
    duplicates = df.duplicated().sum()
    unique = total - duplicates
    
    fig = px.pie(
        names=["Unique Records", "Duplicate Records"],
        values=[unique, duplicates],
        color=["Unique Records", "Duplicate Records"],
        color_discrete_map={"Unique Records": "#10B981", "Duplicate Records": "#EF4444"},
        hole=0.4
    )
    
    fig.update_layout(
        title="Duplicate Analysis",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1F2937", 'family': "Outfit, Inter"},
        margin=dict(l=10, r=10, t=50, b=10),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def plot_datatype_bar(inferred_types: dict) -> go.Figure:
    """
    Plots the count distribution of inferred column data types.
    inferred_types: dict of {column: type_string}
    """
    type_series = pd.Series(list(inferred_types.values()))
    counts = type_series.value_counts().reset_index()
    counts.columns = ["Data Type", "Count"]
    
    # High-quality custom theme colors
    color_map = {
        "Numeric": "#3B82F6",      # Blue
        "Categorical": "#8B5CF6",  # Purple
        "Text": "#EC4899",         # Pink
        "DateTime": "#F59E0B",     # Amber
        "Boolean": "#10B981",      # Emerald
        "Unknown": "#9CA3AF"       # Grey
    }
    
    # Apply color map
    colors = [color_map.get(t, "#6B7280") for t in counts["Data Type"]]
    
    fig = go.Figure(data=[go.Bar(
        x=counts["Data Type"],
        y=counts["Count"],
        marker_color=colors,
        text=counts["Count"],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Data Type Distribution",
        xaxis_title="Inferred Type",
        yaxis_title="Number of Columns",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1F2937", 'family': "Outfit, Inter"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=260,
        yaxis=dict(gridcolor="#E5E7EB")
    )
    
    return fig

def plot_outlier_boxplot(df: pd.DataFrame, col: str) -> go.Figure:
    """
    Generates a boxplot showing numerical data distribution and potential outliers.
    """
    if df is None or df.empty or col not in df.columns:
        return go.Figure()
        
    fig = px.box(
        df,
        y=col,
        points="outliers",
        color_discrete_sequence=["#6366F1"] # Indigo
    )
    
    fig.update_layout(
        title=f"Boxplot for '{col}'",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1F2937", 'family': "Outfit, Inter"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        yaxis=dict(gridcolor="#E5E7EB")
    )
    
    return fig

def plot_outlier_scatter(df: pd.DataFrame, col: str, outlier_mask: pd.Series) -> go.Figure:
    """
    Generates a scatter plot of row index vs column value, highlighting outliers in red.
    """
    if df is None or df.empty or col not in df.columns or outlier_mask is None:
        return go.Figure()
        
    plot_df = pd.DataFrame({
        "Index": df.index,
        "Value": pd.to_numeric(df[col], errors='coerce'),
        "Status": np.where(outlier_mask, "Outlier", "Normal")
    }).dropna(subset=["Value"])
    
    # Sort status to ensure outliers are drawn on top
    plot_df = plot_df.sort_values(by="Status", ascending=False)
    
    fig = px.scatter(
        plot_df,
        x="Index",
        y="Value",
        color="Status",
        color_discrete_map={"Normal": "#3B82F6", "Outlier": "#EF4444"},
        hover_data={"Index": True, "Value": True, "Status": True}
    )
    
    fig.update_layout(
        title=f"Outlier Scatter Plot for '{col}'",
        xaxis_title="Dataset Row Index",
        yaxis_title="Value",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1F2937", 'family': "Outfit, Inter"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        xaxis=dict(gridcolor="#E5E7EB"),
        yaxis=dict(gridcolor="#E5E7EB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig
