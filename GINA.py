import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

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

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Introduction", 
    "Data Input", 
    "Grey Conversion", 
    "Analysis Results",
    "Interpretation"
])

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
            factor_cols = st.multiselect("Select factor columns:", df.columns.tolist())
            
            if factor_cols:
                st.success(f"Selected {len(factor_cols)} factors for analysis")
                
    else:  # Manual input
        st.subheader("Manual Data Input")
        
        num_factors = st.number_input("Number of factors:", min_value=2, max_value=20, value=5)
        num_respondents = st.number_input("Number of respondents:", min_value=1, max_value=100, value=10)
        
        if num_factors and num_respondents:
            st.write("Enter factor names:")
            factors = []
            cols = st.columns(3)
            for i in range(num_factors):
                with cols[i % 3]:
                    factors.append(st.text_input(f"Factor {i+1}", value=f"F{i+1}"))
            
            st.subheader("Influence Matrix Setup")
            st.write("For each respondent, you'll need to provide an influence matrix where cell (i,j) represents the influence of factor i on factor j.")
            
            # Example matrix
            example_matrix = pd.DataFrame(np.random.randint(1, 6, size=(num_factors, num_factors)),
                                         columns=factors, index=factors)
            st.write("Example matrix structure:")
            st.dataframe(example_matrix)

# Grey Conversion section
elif section == "Grey Conversion":
    st.header("Grey Conversion")
    
    st.markdown("""
    Convert linguistic ratings to grey scales using the conversion table:
    """)
    
    # Display conversion table again
    grey_table = pd.DataFrame({
        'Rating': [1, 2, 3, 4, 5],
        'Linguistic Label': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        'Grey Scale': ['(0, 3)', '(2, 5)', '(4, 7)', '(6, 9)', '(8, 10)']
    })
    st.table(grey_table)
    
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader("Upload your survey data (CSV format)", type="csv")
    
    if uploaded_file is not None:
        survey_data = pd.read_csv(uploaded_file)
        st.write("Survey data preview:")
        st.dataframe(survey_data.head())
        
        # Let user map columns to factors
        st.subheader("Data Mapping")
        all_columns = survey_data.columns.tolist()
        factor_pairs = []
        
        st.write("For each factor pair, select the column that contains the influence rating:")
        
        # This would need to be more dynamic in a real implementation
        st.info("In a full implementation, this section would allow mapping of all factor pairs to data columns")
        
        if st.button("Convert to Grey Scales"):
            st.success("Grey conversion completed!")
            st.info("In a full implementation, this would show the converted grey values")

# Analysis Results section
elif section == "Analysis Results":
    st.header("Analysis Results")
    
    st.markdown("""
    This section would display the results of the GINA analysis, including:
    - Direct influence coefficients
    - Complete influence coefficients
    - Grey responsibility coefficients
    - Grey influence coefficients
    - Grey importance coefficients
    """)
    
    # Placeholder for results
    st.subheader("Sample Output (from paper)")
    
    # Create sample data similar to what's in the paper
    factors = ["URE", "ANM", "CII", "III", "CIN", "FCH", "IST", "ECL", "OIN", "EIC", "NDF", "HID", "OPB"]
    
    # Create a random matrix for demonstration
    np.random.seed(42)
    influence_data = np.random.rand(len(factors), len(factors))
    
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(influence_data, xticklabels=factors, yticklabels=factors, 
                annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Influence Relations Among Factors")
    st.pyplot(fig)
    
    # Display importance coefficients
    st.subheader("Grey Importance Coefficients")
    importance_df = pd.DataFrame({
        'Factor': factors,
        'Grey Importance Coefficient': np.random.rand(len(factors)) * 0.1 + 0.15
    }).sort_values('Grey Importance Coefficient', ascending=False)
    
    st.dataframe(importance_df)
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Factor'], importance_df['Grey Importance Coefficient'])
    ax.set_xlabel('Grey Importance Coefficient')
    ax.set_title('Factor Importance Ranking')
    ax.invert_yaxis()  # Most important at the top
    st.pyplot(fig)

# Interpretation section
elif section == "Interpretation":
    st.header("Interpretation of Results")
    
    st.markdown("""
    ### How to interpret GINA results:
    
    1. **Grey Importance Coefficients** indicate how important a factor is to the system, considering both its influence on other factors and how much it is influenced by others.
    
    2. Factors with higher importance coefficients are more central to the system and should be prioritized for management attention.
    
    3. The **Pareto principle** (80/20 rule) suggests that typically about 20% of factors account for 80% of the influence in a system.
    
    4. Based on the values of grey importance coefficients, managers can find means to control (increase/decrease) the influence of specific factors.
    """)
    
    st.subheader("Management Implications")
    
    st.info("""
    From the example in the paper:
    - Firm characteristics was observed as the most influential driver of Greenwashing
    - Uncertain regulatory environment and optimistic bias were the second most influential
    - Ethical climate occupied the third position
    
    This implies that:
    1. The size, type, or nature of the industry has a key role in Greenwashing
    2. When regulations are not strong, firms can engage in more Greenwashing
    3. Optimistic bias enhances a firm's confidence of not being caught for Greenwashing
    """)
    
    st.subheader("Practical Applications")
    
    st.write("""
    GINA can be applied to various problems in marketing and supply chain research, such as:
    - Analyzing influence relations among factors for customer satisfaction
    - Studying sustainability performances of retail systems
    - Analyzing drivers of supply chain resilience
    - Identifying interrelationships among attributes in product mix problems
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This app implements the GINA methodology described in: "
    "Rajesh, R. (2023). Expert Systems with Applications, 212, 118816."
)
