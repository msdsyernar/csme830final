import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="House Price Prediction Project",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏡 House Price Prediction - USA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Complete Data Science Project from EDA to Deployment</div>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
st.header("📚 Project Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This project demonstrates a **complete machine learning pipeline** for predicting house prices 
    in the United States using real estate data from Realtor.com.
    
    ### 🎯 Project Goals
    - Analyze and preprocess real estate data with 2.2M+ records
    - Handle missing values intelligently (MNAR, MAR, MCAR)
    - Engineer meaningful features for better predictions
    - Compare multiple ML models and select the best performer
    - Deploy an interactive price prediction tool
    
    ### 📊 Dataset Information
    - **Source:** Realtor.com
    - **Records:** 2,226,382 listings
    - **Features:** 12 original features
    - **Target:** House price prediction
    """)

with col2:
    st.info("""
    ### 🛠️ Technologies Used
    - **Python** - Programming
    - **Pandas** - Data manipulation
    - **Scikit-learn** - ML models
    - **CatBoost** - Best model
    - **Streamlit** - Web app
    - **Plotly** - Visualizations
    """)

st.markdown("---")

# Project Workflow
st.header("🔄 Project Workflow")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("### 1️⃣ Data Overview")
    st.markdown("""
    - Initial exploration
    - Stratified sampling
    - Missingness report
    - Duplicate detection
    """)

with col2:
    st.markdown("### 2️⃣ Missingness Analysis")
    st.markdown("""
    - MNAR identification
    - Pattern analysis
    - Correlation study
    - Visualizations
    """)

with col3:
    st.markdown("### 3️⃣ Preprocessing")
    st.markdown("""
    - Outlier removal
    - Imputation
    - Feature engineering
    - Encoding
    """)

with col4:
    st.markdown("### 4️⃣ Model Selection")
    st.markdown("""
    - Linear Regression
    - Random Forest
    - CatBoost
    - Cross-validation
    """)

with col5:
    st.markdown("### 5️⃣ Prediction")
    st.markdown("""
    - Interactive tool
    - Price estimation
    - Insights
    - Recommendations
    """)

st.markdown("---")

# Key Findings
st.header("🔍 Key Findings & Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Best Model Performance",
        value="R² = 0.80",
        delta="CatBoost Regressor"
    )
    
with col2:
    st.metric(
        label="Prediction Accuracy",
        value="MAE: $54,892",
        delta="-39% vs Baseline"
    )

with col3:
    st.metric(
        label="Data Processing",
        value="170+ K samples",
        delta="10% stratified sample"
    )

st.markdown("---")

# Key Insights
st.header("💡 Key Insights from Analysis")

insights = [
    {
        "title": "📍 Location is King",
        "description": "City, state, and ZIP code features contribute 56% of the model's predictive power. Location dominates house pricing."
    },
    {
        "title": "🏠 House Size Matters",
        "description": "House size and its logarithmic transformation are the second most important features, contributing 25% to predictions."
    },
    {
        "title": "🔄 Missingness is Informative",
        "description": "31.69% of prev_sold_date missing values indicate new constructions. This missingness pattern (MNAR) carries valuable information."
    },
    {
        "title": "🌆 Urban vs Rural Patterns",
        "description": "Properties without acre_lot data are typically smaller urban condos/apartments. Size and location patterns differ significantly."
    },
    {
        "title": "🎯 Feature Engineering Impact",
        "description": "Engineered features (interactions, ratios, encodings) improved model performance by 15% over baseline features alone."
    },
    {
        "title": "⚖️ Model Selection",
        "description": "CatBoost outperformed Random Forest and Linear Regression with better generalization and lower overfitting."
    }
]

col1, col2 = st.columns(2)

for i, insight in enumerate(insights):
    with col1 if i % 2 == 0 else col2:
        with st.expander(insight["title"]):
            st.markdown(insight["description"])

st.markdown("---")

# Navigation Guide
st.header("🗺️ Navigation Guide")

st.markdown("""
Use the **sidebar** to navigate through different sections of the project:

1. **🏠 Overview** - Initial data exploration and sampling
2. **🔍 Missingness Analysis** - Understanding missing data patterns
3. **🛠️ Data Preprocessing** - Feature engineering and cleaning
4. **🤖 Model Selection** - Comparing ML models and results
5. **🎯 Price Prediction** - Interactive prediction tool
6. **📍  City Visualizer** - Interactive visualizer tool    

Each page demonstrates a critical step in the data science workflow!
""")

st.markdown("---")

# Statistics
st.header("📈 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Original Dataset", "1.7M+ rows")
    st.caption("Full dataset size")

with col2:
    st.metric("Working Dataset", "170K+ rows")
    st.caption("10% stratified sample")

with col3:
    st.metric("Features Created", "14 features")
    st.caption("After engineering")

with col4:
    st.metric("Models Tested", "3 models")
    st.caption("Linear, RF, CatBoost")

st.markdown("---")

# Footer
st.success("✨ **Ready to explore?** Use the sidebar to navigate through the project pages!")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("📚 Project Guide")
    
    st.markdown("""
    ### 🚀 Quick Start
    
    Navigate through the pages in order:
    
    1. **Overview** → Understand the data
    2. **Missingness** → Analyze patterns
    3. **Preprocessing** → See transformations
    4. **Models** → Compare performance
    5. **Prediction** → Try the tool!
    
    ---
    
    ### 📂 Required 
                
    Make sure you are in a good mood!
    ---
    
    ### 🎓 Learning Outcomes
    
    ✅ Data preprocessing techniques  
    ✅ Missing data handling  
    ✅ Feature engineering  
    ✅ Model comparison  
    ✅ ML deployment  
                
    ### Who might be interested in the project?
        -> house buyers
        -> house sellers
        -> real state agency
        -> government organizations to define the price range in regions
    
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Each page includes detailed explanations and visualizations!")