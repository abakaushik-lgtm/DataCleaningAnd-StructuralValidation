# Intelligent Data Cleaning and Structural Validation Platform

An intelligent, interactive, and modular data preprocessing platform built with Python, Streamlit, and Pandera. It automatically profiles, cleans, standardizes, and validates raw datasets (CSV, Excel, JSON) to ensure data quality and schema compliance prior to downstream analytics or machine learning.

---

## 🚀 Features

### 1. Data Import & Preview
- Support for **CSV**, **Excel (.xlsx)**, and **JSON** array datasets.
- Interactive data table previews with file dimensions and metadata display.

### 2. Automated Data Profiling
- Real-time logical type inference (`Numeric`, `Categorical`, `Text`, `DateTime`, `Boolean`).
- Column completeness percentages and unique value ratios.
- Statistical descriptions (mean, median, range, std dev) for numeric data.
- Top frequency value distributions for categorical columns.

### 3. Intelligent Data Cleaning
- **Missing Value Imputation**: Supports mean, median, mode, constant (custom) fills, as well as row/column dropping.
- **Duplicate Records Removal**: Exact and partial (subset key columns) duplication filters with first/last occurrence controls.
- **Data Standardization**:
  - Name casing formatter (e.g. `john DOE` $\rightarrow$ `John Doe`).
  - Dates parser to standard `YYYY-MM-DD` format.
  - Phone numbers normalization (e.g. `5551234567` $\rightarrow$ `+1-555-123-4567`).
  - Whitespace cleaning (trimming and collapsing duplicate spacing).
  - Standard text casing operations (UPPERCASE/lowercase).

### 4. Structural Schema & Rules Engine
- **Schema Validation**: Define expected column types dynamically (Integer, Float, String, Date, Boolean) and catch type violations.
- **Custom Logic Rules**: Configure constraint checks such as ranges (`Age >= 18`), string lengths (`Phone length == 10`), format matches (`Email is_email`), and null barriers.
- **Outlier Flagging**: Statistical anomaly detection using both Z-Score and Interquartile Range (IQR) methods.

### 5. High-Fidelity Dashboard
- **Completeness Heatmap**: Visualizes missing value gaps across columns.
- **Quality Score Gauge**: Renders overall score based on completeness, uniqueness, validity, and consistency dimensions.
- **Duplicates Pie Chart**: Displays proportion of duplicate vs unique records.
- **Data Type Distribution Bar Chart**: Breaks down the number of columns of each logical type.
- **Outlier Box & Scatter Plots**: Interactive outlier charts highlighting anomalous rows.

### 6. Cleaning Reports & Exports
- Export cleaned datasets to CSV, Excel (.xlsx), or JSON.
- Download a professional, automated PDF quality report summarizing rows/columns shifts, improvements in the quality score, and details of all cleaning modifications.

---

## 🛠 Technology Stack

- **Data Processing**: Pandas, NumPy
- **Validation**: Pandera
- **Visualization**: Plotly
- **PDF Report Generation**: fpdf2
- **Dashboard UI**: Streamlit
- **Testing**: Pytest

---

## 📂 Directory Structure

```
.
├── app.py                      # Streamlit dashboard layout and UI logic
├── requirements.txt            # Package dependencies configuration
├── dirty_dataset.csv           # Sample dirty CSV for manual testing
├── src/
│   ├── __init__.py
│   ├── profiler.py             # Feature stats & data type inferences
│   ├── cleaner.py              # Null filters, imputers, and standardizers
│   ├── validator.py            # Schema validations, rules engine, and scoring
│   ├── visualizer.py           # Plotly gauge, heatmap, and outlier chart builders
│   └── reporter.py             # PDF report compilers & byte converters
└── tests/
    ├── test_cleaner.py         # Data cleaning unit tests
    └── test_validator.py       # Data validation unit tests
```

---

## ⚙ Installation & Quickstart

### Prerequisites
Make sure you have Python (version $\ge$ 3.8) and Git installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/abakaushik-lgtm/DataCleaningAnd-StructuralValidation.git
cd DataCleaningAnd-StructuralValidation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
This will start the Streamlit server locally. Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the platform.

### 4. Interactive Sandbox Testing
We provide a **`dirty_dataset.csv`** file in the root directory. Upload it into the dashboard using the sidebar file loader to immediately test all profiling, cleaning, validation, and dashboard features.

---

## 🧪 Running Tests

Unit tests are included under the `tests/` directory. Run them using:
```bash
python -m pytest
```
All tests should pass, showing green checkmarks.
