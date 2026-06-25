import pandas as pd
import numpy as np
import io
from fpdf import FPDF
from datetime import datetime

class DataQualityReport(FPDF):
    def header(self):
        # Header background banner
        self.set_fill_color(31, 41, 55) # Dark Grey/Slate
        self.rect(0, 0, 210, 30, 'F')
        
        # Header text
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, 'DATA CLEANING & QUALITY REPORT', align='C', border=0)
        self.ln(10)
        self.set_font('Helvetica', '', 10)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='C', border=0)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(156, 163, 175) # Light grey
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(
    summary_stats: dict, 
    cleaning_logs: list, 
    validation_summary: dict,
    initial_score: float, 
    final_score: float
) -> bytes:
    """
    Generates a PDF report containing before-and-after data comparisons, 
    quality score upgrades, and details of all cleaning operations.
    """
    # Create FPDF instance
    pdf = DataQualityReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # Margin
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    
    pdf.ln(5)
    
    # 1. Executive Summary Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 8, '1. Executive Summary', border=0)
    pdf.ln(10)
    
    # Metrics Box (Quality score and dimensions comparison)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(243, 244, 246) # Light grey background
    pdf.cell(60, 8, ' Metric', border=1, fill=True)
    pdf.cell(60, 8, ' Before Cleaning', border=1, fill=True)
    pdf.cell(60, 8, ' After Cleaning', border=1, fill=True)
    pdf.ln(8)
    
    pdf.set_font('Helvetica', '', 10)
    # Score Row
    pdf.cell(60, 8, ' Data Quality Score', border=1)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(185, 28, 28) # Red for initial
    pdf.cell(60, 8, f' {initial_score:.1f}%', border=1)
    pdf.set_text_color(4, 120, 87) # Green for final
    pdf.cell(60, 8, f' {final_score:.1f}%', border=1)
    pdf.set_text_color(31, 41, 55)
    pdf.ln(8)
    
    # Rows Count
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(60, 8, ' Rows Count', border=1)
    pdf.cell(60, 8, f' {summary_stats.get("initial_rows", 0)}', border=1)
    pdf.cell(60, 8, f' {summary_stats.get("final_rows", 0)}', border=1)
    pdf.ln(8)
    
    # Columns Count
    pdf.cell(60, 8, ' Columns Count', border=1)
    pdf.cell(60, 8, f' {summary_stats.get("initial_cols", 0)}', border=1)
    pdf.cell(60, 8, f' {summary_stats.get("final_cols", 0)}', border=1)
    pdf.ln(8)
    
    # Duplicates count
    pdf.cell(60, 8, ' Duplicate Rows', border=1)
    pdf.cell(60, 8, f' {summary_stats.get("initial_duplicates", 0)}', border=1)
    pdf.cell(60, 8, f' {summary_stats.get("final_duplicates", 0)}', border=1)
    pdf.ln(12)
    
    # 2. Data Cleaning Log Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, '2. Detailed Cleaning Operations Log', border=0)
    pdf.ln(10)
    
    if len(cleaning_logs) == 0:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(0, 8, 'No cleaning operations were performed on the dataset.', border=0)
        pdf.set_text_color(31, 41, 55)
        pdf.ln(10)
    else:
        pdf.set_font('Helvetica', '', 10)
        for i, log in enumerate(cleaning_logs, 1):
            # Bullet point output
            log_text = f"{i}. {log}"
            # Multi_cell handles word wrapping nicely
            pdf.multi_cell(0, 6, log_text, border=0)
            pdf.ln(2)
        pdf.ln(6)
        
    # 3. Structural Validation Summary Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, '3. Structural & Rules Validation Status', border=0)
    pdf.ln(10)
    
    if not validation_summary:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(0, 8, 'No structural schema validation or custom rules were configured.', border=0)
        pdf.ln(10)
    else:
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(31, 41, 55)
        
        # Schema status
        is_schema_valid = validation_summary.get("schema_valid", True)
        schema_status = "PASSED" if is_schema_valid else "FAILED / MISMATCH DETECTED"
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(50, 6, "Schema Consistency:", border=0)
        if is_schema_valid:
            pdf.set_text_color(4, 120, 87) # Green
        else:
            pdf.set_text_color(185, 28, 28) # Red
        pdf.cell(0, 6, schema_status, border=0)
        pdf.set_text_color(31, 41, 55)
        pdf.ln(6)
        
        # Rules status
        rules_applied = validation_summary.get("total_rules", 0)
        rules_failed = validation_summary.get("failed_rules", 0)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(50, 6, "Custom Rules Validation:", border=0)
        pdf.set_font('Helvetica', '', 10)
        
        if rules_applied == 0:
            pdf.cell(0, 6, "No custom rules applied.", border=0)
        elif rules_failed == 0:
            pdf.set_text_color(4, 120, 87)
            pdf.cell(0, 6, f"PASSED (All {rules_applied} rules met)", border=0)
        else:
            pdf.set_text_color(185, 28, 28)
            pdf.cell(0, 6, f"FAILED ({rules_failed} out of {rules_applied} rules failed)", border=0)
            
        pdf.set_text_color(31, 41, 55)
        pdf.ln(12)
        
        # Details of mismatches/failures
        errors = validation_summary.get("errors", [])
        if errors:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Validation Errors Detail (Top 10)', border=0)
            pdf.ln(8)
            
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_fill_color(243, 244, 246)
            pdf.cell(30, 6, ' Column', border=1, fill=True)
            pdf.cell(20, 6, ' Row Index', border=1, fill=True)
            pdf.cell(40, 6, ' Error Type', border=1, fill=True)
            pdf.cell(90, 6, ' Description', border=1, fill=True)
            pdf.ln(6)
            
            pdf.set_font('Helvetica', '', 9)
            for err in errors[:10]:
                col_name = str(err.get('column', ''))
                row_idx = str(err.get('row_index', ''))
                err_type = str(err.get('error_type', ''))
                desc = str(err.get('description', ''))
                
                # Check line height and draw
                pdf.cell(30, 6, f" {col_name[:14]}", border=1)
                pdf.cell(20, 6, f" {row_idx}", border=1)
                pdf.cell(40, 6, f" {err_type}", border=1)
                pdf.cell(90, 6, f" {desc[:48]}", border=1)
                pdf.ln(6)
                
    # Output byte string
    pdf_bytes = pdf.output()
    # fpdf2 output() returns a bytearray or string depending on version, let's cast to bytes
    return bytes(pdf_bytes)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Converts a dataframe to CSV bytes for download.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode('utf-8')

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Converts a dataframe to Excel bytes for download using openpyxl.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned Data')
    return buffer.getvalue()

def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    """
    Converts a dataframe to JSON bytes for download.
    """
    buffer = io.StringIO()
    df.to_json(buffer, orient='records', indent=2)
    return buffer.getvalue().encode('utf-8')
