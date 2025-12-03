import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Data Overview", page_icon="🏠", layout="wide")

st.title(" House Price Prediction - USA")
st.markdown("---")

# Dataset Information
st.header("📊 About Dataset")
st.markdown("""
This dataset contains Real Estate listings in the US broken by State and zip code.

**Original Dataset Features:**
- `brokered_by`: Categorically encoded agency/broker
- `status`: Housing status (ready for sale or ready to build)
- `price`: Housing price (current listing or recently sold)
- `bed`: Number of beds
- `bath`: Number of bathrooms
- `acre_lot`: Property/Land size in acres
- `street`: Categorically encoded street address
- `city`: City name
- `state`: State name
- `zip_code`: Postal code
- `house_size`: House area/size in square feet
- `prev_sold_date`: Previously sold date

**Data Source:** [Realtor.com](https://www.realtor.com/)
""")
 
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        # data = pd.read_csv(r'c:\Users\Администратор\Documents\study\CSME 830\house_price_project\data\raw\raw_data.csv')
        data = pd.read_csv(r'data\raw\raw_data.csv')
        return data
    except:
        st.error("⚠️ Please ensure 'train_set.csv' is in the same directory")
        return None

data = load_data()

if data is not None:
    st.header("🔍 Initial Data Exploration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{data.shape[0]:,}")
    with col2:
        st.metric("Features", data.shape[1])
    with col3:
        st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Display first few rows
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Basic statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Initial Data Selection
    st.header("🎯 Initial Data Selection")
    st.markdown("""
    **Columns Dropped:**
    - `brokered_by` and `street` were encoded due to data privacy policy
    - These columns provide no useful information for prediction
    
    **Price Handling:**
    - Dropped rows where `price` is NaN (target variable cannot be missing)
    """)
    
    # Processing steps
    with st.spinner("Processing data..."):
        # Drop columns
        data_processed = data.drop(columns=['brokered_by', 'street'], errors='ignore')
        
        # Drop NaN in price
        initial_shape = data_processed.shape[0]
        data_processed = data_processed.dropna(subset=['price'])
        rows_dropped = initial_shape - data_processed.shape[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"✅ Rows before dropping price NaN: **{initial_shape:,}**")
        with col2:
            st.info(f"✅ Rows after dropping price NaN: **{data_processed.shape[0]:,}** (Dropped: {rows_dropped:,})")
    
    st.markdown("---")
    
    # Stratified Sampling
    st.header("🎲 Stratified Sampling")
    st.markdown("""
    To ensure representative distribution across price ranges, we use **stratified sampling**:
    - Divide prices into 10 bins
    - Sample 10% from each bin
    - Maintains price distribution in smaller dataset
    """)
    
    # Create bins and sample
    data_processed['price_bins'] = pd.cut(data_processed['price'], bins=10, labels=False)
    freqs = data_processed['price_bins'].value_counts()
    data_processed = data_processed[data_processed['price_bins'].isin(freqs[freqs >= 2].index)]
    
    _, sample_data = train_test_split(
        data_processed, 
        test_size=0.1, 
        stratify=data_processed['price_bins'], 
        random_state=42
    )
    
    sample_data = sample_data.drop(columns=['price_bins'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"📊 Original Size: **{data_processed.shape[0]:,}** rows")
    with col2:
        st.success(f"📊 Sample Size: **{sample_data.shape[0]:,}** rows (10%)")
    
    # Memory improvement
    memory_before = data_processed.memory_usage(deep=True).sum() / 1024**2
    memory_after = sample_data.memory_usage(deep=True).sum() / 1024**2
    st.info(f"💾 Memory reduced from **{memory_before:.1f} MB** to **{memory_after:.1f} MB**")
    
    st.markdown("---")
    
    # Price Distribution
    st.header("💰 Price Distribution Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(np.log1p(sample_data['price']), bins=50, stat='density', color="skyblue", ax=ax)
    sns.kdeplot(np.log1p(sample_data['price']), color="red", linewidth=2, ax=ax)
    ax.set_title("Log-Transformed House Price Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("log(price)")
    ax.set_ylabel("Density")
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Missingness Report
    st.header("🔎 Initial Missingness Report")
    
    def missing_report(df):
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        missing_data = pd.DataFrame({
            'Missing Values': missing_values, 
            'Percentage': missing_percentage.round(2)
        })
        return missing_data[missing_data['Missing Values'] > 0].sort_values(by="Percentage", ascending=False)
    
    missing_df = missing_report(sample_data)
    
    if len(missing_df) > 0:
        st.dataframe(missing_df, use_container_width=True)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        missing_df['Percentage'].plot(kind='barh', color='coral', ax=ax)
        ax.set_xlabel('Percentage (%)')
        ax.set_title('Missing Data by Feature')
        st.pyplot(fig)
    else:
        st.success("✅ No missing values found!")
    
    st.markdown("---")
    
    # Duplicates
    st.header("🔄 Duplicate Records")
    dup_count = sample_data.duplicated().sum()
    
    if dup_count > 0:
        st.warning(f"⚠️ Found **{dup_count:,}** duplicate records")
        sample_data = sample_data.drop_duplicates()
        st.success(f"✅ Duplicates removed. New shape: **{sample_data.shape}**")
    else:
        st.success("✅ No duplicate records found!")
    
    # Drop minimal missing values
    st.subheader("🧹 Handling Minimal Missing Values")
    st.markdown("""
    **Strategy:** Drop rows with missing values in critical columns:
    - `zip_code`, `state`, `city`: Location is essential
    - `price`: Cannot predict without target variable
    """)
    
    before_drop = sample_data.shape[0]
    sample_data = sample_data.dropna(subset=['zip_code', 'state', 'city', 'price'])
    after_drop = sample_data.shape[0]
    
    st.info(f"Rows dropped: **{before_drop - after_drop:,}**")
    
    # Final missing report
    st.subheader("📊 Updated Missingness Report")
    final_missing = missing_report(sample_data)
    if len(final_missing) > 0:
        st.dataframe(final_missing, use_container_width=True)
    else:
        st.success("✅ All minimal missing values handled!")

st.markdown("---")
st.markdown("### 👉 Next: Explore **Missingness Analysis & Visualizations** in the next page")