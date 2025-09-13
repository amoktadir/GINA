import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Grey Influence Analysis (GINA)",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Grey Influence Analysis (GINA)")
st.markdown("""
This application implements the Grey Influence Analysis (GINA) methodology as described in:
*Rajesh, R. (2023). An introduction to grey influence analysis (GINA): Applications to causal modelling in marketing and supply chain research. Expert Systems with Applications, 212, 118816.*

GINA is used to analyze influence relations among factors when there are large numbers of responses, particularly in survey-based or expert response-based studies.
""")

# Grey conversion table
GREY_SCALES = {
    1: {"label": "Very Low", "bounds": (0, 3)},
    2: {"label": "Low", "bounds": (2, 5)},
    3: {"label": "Medium", "bounds": (4, 7)},
    4: {"label": "High", "bounds": (6, 9)},
    5: {"label": "Very High", "bounds": (8, 10)}
}

# GINA calculation functions
def convert_to_grey(rating):
    """Convert a rating to grey scale bounds"""
    return GREY_SCALES[rating]["bounds"]

def aggregate_grey_responses(responses):
    """Aggregate grey responses using grey addition"""
    lower_sum = sum(r[0] for r in responses)
    upper_sum = sum(r[1] for r in responses)
    return (lower_sum, upper_sum)

def whiten_grey_values(grey_value, model="typical"):
    """Whiten grey values to crisp scores"""
    lower, upper = grey_value
    if model == "critical":
        return lower
    elif model == "ideal":
        return upper
    else:  # typical
        return (lower + upper) / 2

def calculate_direct_influence(matrix):
    """Calculate direct influence coefficients"""
    col_sums = matrix.sum(axis=0)
    return matrix / col_sums

def calculate_complete_influence(direct_matrix):
    """Calculate complete influence coefficients"""
    identity = np.eye(direct_matrix.shape[0])
    return np.linalg.inv(identity - direct_matrix) - identity

def calculate_responsibility_coefficients(complete_matrix):
    """Calculate grey responsibility coefficients"""
    col_sums = complete_matrix.sum(axis=0)
    total_sum = complete_matrix.sum()
    n = complete_matrix.shape[0]
    return col_sums / (total_sum / n)

def calculate_influence_coefficients(complete_matrix):
    """Calculate grey influence coefficients"""
    row_sums = complete_matrix.sum(axis=1)
    total_sum = complete_matrix.sum()
    n = complete_matrix.shape[0]
    return row_sums / (total_sum / n)

def calculate_importance_coefficients(resp_coeffs, inf_coeffs):
    """Calculate grey importance coefficients"""
    return resp_coeffs + inf_coeffs

# PDF generation function
def create_pdf_report(factors, results, analysis_name):
    """Create a PDF report of the GINA analysis"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"GINA Analysis Report: {analysis_name}", 0, 1, "C")
    pdf.ln(10)
    
    # Date and details
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    
    # Factors
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Factors Analyzed:", 0, 1)
    pdf.set_font("Arial", "", 12)
    factors_text = ", ".join(factors)
    pdf.multi_cell(0, 10, factors_text)
    pdf.ln(5)
    
    # Results summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Key Results:", 0, 1)
    pdf.set_font("Arial", "", 12)
    
    # Most influential factors
    importance_df = results["importance_df"]
    top_factors = importance_df.nlargest(3, "Grey Importance Coefficient")
    
    pdf.cell(0, 10, "Top 3 Most Influential Factors:", 0, 1)
    for i, (_, row) in enumerate(top_factors.iterrows(), 1):
        pdf.cell(0, 10, f"{i}. {row['Factor']} (Score: {row['Grey Importance Coefficient']:.4f})", 0, 1)
    
    pdf.ln(10)
    
    # Complete results table
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Complete Results:", 0, 1)
    
    # Create table
    col_widths = [60, 40, 40, 60]
    pdf.set_font("Arial", "B", 12)
    pdf.cell(col_widths[0], 10, "Factor", 1)
    pdf.cell(col_widths[1], 10, "Responsibility", 1)
    pdf.cell(col_widths[2], 10, "Influence", 1)
    pdf.cell(col_widths[3], 10, "Importance", 1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 12)
    for _, row in importance_df.iterrows():
        pdf.cell(col_widths[0], 10, row["Factor"], 1)
        pdf.cell(col_widths[1], 10, f"{row['Grey Responsibility Coefficient']:.4f}", 1)
        pdf.cell(col_widths[2], 10, f"{row['Grey Influence Coefficient']:.4f}", 1)
        pdf.cell(col_widths[3], 10, f"{row['Grey Importance Coefficient']:.4f}", 1)
        pdf.ln()
    
    # Save to bytes buffer
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    
    return buffer

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Introduction", 
    "Data Input", 
    "Analysis Configuration",
    "Results",
    "PDF Report"
])

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = None
if "factors" not in st.session_state:
    st.session_state.factors = None

# Introduction section
if section == "Introduction":
    st.header("Introduction to GINA Methodology")
    
    st.markdown("""
    ### Key Steps in GINA:
    1. **Identify factors** for study through literature review, expert opinions, etc.
    2. **Obtain influence relations** among factors through surveys or expert responses
    3. **Convert responses to grey scales** using a conversion table
    4. **Aggregate influence relations** in grey scales
    5. **Whiten grey values** into crisp scores using three models:
       - Critical model (lower bound values)
       - Ideal model (upper bound values)
       - Typical model (average of bounds)
    6. **Calculate direct influence coefficients**
    7. **Obtain complete influence coefficients**
    8. **Calculate grey responsibility and influence coefficients**
    9. **Compute grey importance coefficients**
    10. **Identify most influential factors**
    """)
    
    st.subheader("Grey Scale Conversion Table")
    grey_table = pd.DataFrame({
        'Rating': [1, 2, 3, 4, 5],
        'Linguistic Label': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        'Grey Scale': ['(0, 3)', '(2, 5)', '(4, 7)', '(6, 9)', '(8, 10)'],
        'Lower Bound': [0, 2, 4, 6, 8],
        'Upper Bound': [3, 5, 7, 9, 10]
    })
    st.table(grey_table)

# Data Input section
elif section == "Data Input":
    st.header("Data Input")
    
    input_method = st.radio("Select input method:", 
                          ["Upload CSV file", "Manual input"])
    
    if input_method == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.dataframe(df)
            
            # Let user select which columns contain the influence ratings
            st.subheader("Data Configuration")
            all_columns = df.columns.tolist()
            factor_cols = st.multiselect("Select factor columns:", all_columns)
            
            if factor_cols:
                st.session_state.factors = factor_cols
                st.success(f"Selected {len(factor_cols)} factors for analysis")
                
                # Let user specify respondent ID column (optional)
                id_col = st.selectbox("Select respondent ID column (optional):", 
                                    [None] + [col for col in all_columns if col not in factor_cols])
                
                # Store the data in session state
                st.session_state.raw_data = df
                st.session_state.factor_cols = factor_cols
                st.session_state.id_col = id_col
                
    else:  # Manual input
        st.subheader("Manual Data Input")
        
        analysis_name = st.text_input("Analysis Name:", value="GINA Analysis")
        num_factors = st.number_input("Number of factors:", min_value=2, max_value=20, value=5)
        num_respondents = st.number_input("Number of respondents:", min_value=1, max_value=100, value=10)
        
        if num_factors and num_respondents:
            st.write("Enter factor names:")
            factors = []
            cols = st.columns(3)
            for i in range(num_factors):
                with cols[i % 3]:
                    factors.append(st.text_input(f"Factor {i+1}", value=f"F{i+1}"))
            
            st.session_state.factors = factors
            
            st.subheader("Influence Matrix Setup")
            st.write("For each respondent, provide influence ratings between factors (1-5 scale)")
            
            # Create empty data structure
            data = {factor: [] for factor in factors}
            data['Respondent'] = []
            
            # Create input matrix for each respondent
            for resp in range(num_respondents):
                st.markdown(f"### Respondent {resp+1}")
                resp_data = {}
                
                # Create a matrix input
                cols = st.columns(num_factors + 1)
                with cols[0]:
                    st.write("Influences→")
                    st.write("Influenced↓")
                
                for i, factor in enumerate(factors):
                    with cols[i+1]:
                        st.write(factor)
                
                for i, influenced in enumerate(factors):
                    row_cols = st.columns(num_factors + 1)
                    with row_cols[0]:
                        st.write(influenced)
                    for j, influencer in enumerate(factors):
                        with row_cols[j+1]:
                            rating = st.slider(
                                f"{influencer}→{influenced}",
                                min_value=1, max_value=5, value=3,
                                key=f"resp{resp}_{i}_{j}"
                            )
                            resp_data[f"{influencer}→{influenced}"] = rating
                
                # Store respondent data
                for key, value in resp_data.items():
                    data[key].append(value)
                data['Respondent'].append(f"R{resp+1}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            st.session_state.raw_data = df
            st.session_state.factor_cols = [f"{inf}→{infl}" for inf in factors for infl in factors]
            st.session_state.id_col = "Respondent"

# Analysis Configuration section
elif section == "Analysis Configuration":
    st.header("Analysis Configuration")
    
    if "raw_data" not in st.session_state:
        st.warning("Please input data first in the 'Data Input' section.")
    else:
        st.success("Data loaded successfully!")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.raw_data.head())
        
        # Analysis settings
        st.subheader("Analysis Settings")
        whitening_model = st.selectbox(
            "Select whitening model:",
            ["typical", "critical", "ideal"],
            help="Critical model uses lower bounds, ideal uses upper bounds, typical uses averages"
        )
        
        if st.button("Run GINA Analysis"):
            with st.spinner("Analyzing data..."):
                # Get data
                df = st.session_state.raw_data
                factor_pairs = st.session_state.factor_cols
                
                # Convert ratings to grey scales
                grey_data = {}
                for pair in factor_pairs:
                    grey_values = [convert_to_grey(rating) for rating in df[pair]]
                    grey_data[pair] = grey_values
                
                # Aggregate grey responses
                aggregated = {}
                for pair, values in grey_data.items():
                    aggregated[pair] = aggregate_grey_responses(values)
                
                # Whiten grey values
                whitened = {}
                for pair, grey_value in aggregated.items():
                    whitened[pair] = whiten_grey_values(grey_value, whitening_model)
                
                # Create influence matrix
                n = len(st.session_state.factors)
                influence_matrix = np.zeros((n, n))
                
                factors = st.session_state.factors
                for i, influencer in enumerate(factors):
                    for j, influenced in enumerate(factors):
                        pair = f"{influencer}→{influenced}"
                        influence_matrix[j, i] = whitened[pair]  # j,i because we want column j to be influenced by row i
                
                # Calculate direct influence coefficients
                direct_matrix = calculate_direct_influence(influence_matrix)
                
                # Calculate complete influence coefficients
                complete_matrix = calculate_complete_influence(direct_matrix)
                
                # Calculate responsibility and influence coefficients
                resp_coeffs = calculate_responsibility_coefficients(complete_matrix)
                inf_coeffs = calculate_influence_coefficients(complete_matrix)
                
                # Calculate importance coefficients
                importance_coeffs = calculate_importance_coefficients(resp_coeffs, inf_coeffs)
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    "Factor": factors,
                    "Grey Responsibility Coefficient": resp_coeffs,
                    "Grey Influence Coefficient": inf_coeffs,
                    "Grey Importance Coefficient": importance_coeffs
                }).sort_values("Grey Importance Coefficient", ascending=False)
                
                # Store results in session state
                st.session_state.results = {
                    "influence_matrix": influence_matrix,
                    "direct_matrix": direct_matrix,
                    "complete_matrix": complete_matrix,
                    "responsibility_coefficients": resp_coeffs,
                    "influence_coefficients": inf_coeffs,
                    "importance_coefficients": importance_coeffs,
                    "importance_df": results_df,
                    "whitening_model": whitening_model
                }
                
                st.success("Analysis completed!")

# Results section
elif section == "Results":
    st.header("Analysis Results")
    
    if st.session_state.results is None:
        st.warning("Please run the analysis first in the 'Analysis Configuration' section.")
    else:
        results = st.session_state.results
        factors = st.session_state.factors
        
        st.subheader(f"Results using {results['whitening_model']} whitening model")
        
        # Display importance coefficients
        st.markdown("### Factor Importance Ranking")
        st.dataframe(results["importance_df"])
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        df_sorted = results["importance_df"].sort_values("Grey Importance Coefficient", ascending=True)
        ax.barh(df_sorted["Factor"], df_sorted["Grey Importance Coefficient"])
        ax.set_xlabel('Grey Importance Coefficient')
        ax.set_title('Factor Importance Ranking')
        st.pyplot(fig)
        
        # Display influence matrix
        st.markdown("### Influence Matrix")
        influence_df = pd.DataFrame(
            results["influence_matrix"],
            columns=factors,
            index=factors
        )
        st.dataframe(influence_df)
        
        # Create a heatmap
        st.markdown("### Influence Relations Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(influence_df, xticklabels=factors, yticklabels=factors, 
                    annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
        ax.set_title("Influence Relations Among Factors")
        st.pyplot(fig)
        
        # Display complete influence matrix
        st.markdown("### Complete Influence Matrix")
        complete_df = pd.DataFrame(
            results["complete_matrix"],
            columns=factors,
            index=factors
        )
        st.dataframe(complete_df)
        
        # Interpretation
        st.markdown("### Interpretation")
        top_factors = results["importance_df"].nlargest(3, "Grey Importance Coefficient")
        
        st.info(f"""
        Based on the GINA analysis, the most influential factors are:
        
        1. **{top_factors.iloc[0]['Factor']}** (Importance score: {top_factors.iloc[0]['Grey Importance Coefficient']:.4f})
        2. **{top_factors.iloc[1]['Factor']}** (Importance score: {top_factors.iloc[1]['Grey Importance Coefficient']:.4f})
        3. **{top_factors.iloc[2]['Factor']}** (Importance score: {top_factors.iloc[2]['Grey Importance Coefficient']:.4f})
        
        These factors should be prioritized for management attention as they have the greatest overall influence
        on the system, both influencing other factors and being influenced by them.
        """)

# PDF Report section
elif section == "PDF Report":
    st.header("PDF Report Generation")
    
    if st.session_state.results is None:
        st.warning("Please run the analysis first to generate a report.")
    else:
        st.info("Generate a PDF report of your GINA analysis results.")
        
        analysis_name = st.text_input("Report Title:", value="GINA Analysis Report")
        
        if st.button("Generate PDF Report"):
            with st.spinner("Generating report..."):
                # Create PDF
                pdf_buffer = create_pdf_report(
                    st.session_state.factors, 
                    st.session_state.results, 
                    analysis_name
                )
                
                # Create download link
                st.success("Report generated successfully!")
                b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="gina_analysis_report.pdf">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This app implements the GINA methodology described in: "
    "Rajesh, R. (2023). Expert Systems with Applications, 212, 118816."
)
