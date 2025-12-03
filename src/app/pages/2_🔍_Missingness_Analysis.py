import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Missingness Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Missingness Analysis & Visualization")
st.markdown("---")

# Load cleaned data
@st.cache_data
def load_data():
    try:
        # data = pd.read_csv(r'c:\Users\Администратор\Documents\study\CSME 830\house_price_project\data\raw\sample_before_cleaning.csv')
        data = pd.read_csv(r'data/raw/sample_before_cleaning.csv')
        return data
    except:
        st.error("⚠️ Please ensure 'sample_before_cleaning.csv' is in the same directory")
        return None

df = load_data()

if df is not None:
    st.header("📊 Understanding Missing Data Patterns")
    
    # prev_sold_date Analysis
    st.subheader("🏘️ 1. prev_sold_date - Missing Not At Random (MNAR)")
    
    st.markdown("""
    **Key Question:** Why is `prev_sold_date` missing in 31.69% of records?
    
    **Possible Reasons:**
    1. Houses never sold before (new constructions)
    2. First-time listings
    3. Data not available
    """)
    
    # Create has_prev_sale feature
    df['has_prev_sale'] = np.where(df['prev_sold_date'].isnull(), 0, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Status Distribution**")
        status_counts = df['status'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        status_counts.plot(kind='bar', color=['skyblue', 'coral'], ax=ax1)
        ax1.set_title('Property Status Count')
        ax1.set_ylabel('Count')
        ax1.set_xlabel('Status')
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    
    with col2:
        st.markdown("**Missingness by Status**")
        status_missing = df.groupby('status')['has_prev_sale'].value_counts(normalize=True).unstack()
        st.dataframe(status_missing, use_container_width=True)
    
    st.info("""
    **Conclusion:** 
    - 50.4% of 'for_sale' properties have no previous sale date
    - 100% of 'ready_to_build' properties have no previous sale date
    - **This is MNAR** - missingness depends on the feature itself (new properties)
    """)
    
    st.success("✅ **Solution:** Created `has_prev_sale` binary feature (1 = previously sold, 0 = new/no record)")
    
    st.markdown("---")
    
    # bed, bath, house_size Analysis
    st.subheader("🏠 2. Bed, Bath, House_size - Pattern Analysis")
    
    st.markdown("""
    **Key Question:** Why do bed, bath, and house_size have 20%+ missing values?
    Are they missing together?
    """)
    
    # Overlap analysis
    bed_missing = df['bed'].isnull()
    bath_missing = df['bath'].isnull()
    house_size_missing = df['house_size'].isnull()
    
    combinations = {
        'None missing': (~bed_missing & ~bath_missing & ~house_size_missing).sum(),
        'Only bed missing': (bed_missing & ~bath_missing & ~house_size_missing).sum(),
        'Only bath missing': (~bed_missing & bath_missing & ~house_size_missing).sum(),
        'Only house_size missing': (~bed_missing & ~bath_missing & house_size_missing).sum(),
        'Bed and bath missing': (bed_missing & bath_missing & ~house_size_missing).sum(),
        'Bed and house_size missing': (bed_missing & ~bath_missing & house_size_missing).sum(),
        'Bath and house_size missing': (~bed_missing & bath_missing & house_size_missing).sum(),
        'All missing': (bed_missing & bath_missing & house_size_missing).sum()
    }
    
    colors_detailed = ['#2ecc71', '#f39c12', '#e67e22', '#3498db', 
                       '#e67e22', '#c0392b', '#e74c3c', '#8e44ad']
    
    fig2 = go.Figure(data=[go.Pie(
        labels=list(combinations.keys()),
        values=list(combinations.values()),
        marker=dict(colors=colors_detailed),
        textinfo='label+percent',
        textposition='auto',
        hole=0.4,
        pull=[0, 0, 0, 0, 0.05, 0.05, 0.05, 0.15]
    )])
    
    fig2.update_layout(
        title={
            'text': '<b>Missing Value Pattern Distribution</b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.warning("⚠️ **Key Finding:** 90%+ of rows with missing house_size also have missing bed and bath")
    
    # Correlation heatmap
    st.subheader("🔥 Missing Value Correlation Matrix")
    
    critical_cols = ['bed', 'bath', 'house_size', 'acre_lot', 'prev_sold_date']
    missing_indicators = df[critical_cols].isnull().astype(int)
    corr_matrix = missing_indicators.corr()
    
    fig3 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdYlGn_r',
        zmid=0.5,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig3.update_layout(
        title='Missing Value Correlation',
        xaxis_title='Features',
        yaxis_title='Features',
        height=500
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Status visualization
    st.subheader("📊 House Size Missingness by Status")
    
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='status', hue=df['house_size'].isnull(), ax=ax4)
    ax4.set_title('House Size Missingness by Property Status')
    ax4.set_xlabel('Status')
    ax4.set_ylabel('Count')
    ax4.legend(title='house_size missing', labels=['False', 'True'])
    st.pyplot(fig4)
    
    st.info("""
    **Conclusion:**
    - 24% of missing values are from 'for_sale' or 'sold' properties
    - These are likely **land-only listings** (no buildings yet)
    - Decision: Drop these rows as we're predicting house prices, not land prices
    """)
    
    st.markdown("---")
    
    # acre_lot Analysis
    st.subheader("🌳 3. Acre_lot - Urban vs Rural Analysis")
    
    st.markdown("""
    **Key Question:** Why is acre_lot missing? Could it be urban apartments/condos?
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with_lot = df[df['acre_lot'].notna()]['house_size'].mean()
        without_lot = df[df['acre_lot'].isna()]['house_size'].mean()
        
        st.metric("Avg House Size WITH lot", f"{with_lot:,.0f} sqft")
        st.metric("Avg House Size WITHOUT lot", f"{without_lot:,.0f} sqft")
    
    with col2:
        with_lot_price = df[df['acre_lot'].notna()]['price'].mean()
        without_lot_price = df[df['acre_lot'].isna()]['price'].mean()
        
        st.metric("Avg Price WITH lot", f"${with_lot_price:,.0f}")
        st.metric("Avg Price WITHOUT lot", f"${without_lot_price:,.0f}")
    
    # Box plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sns.boxplot(
            x=df['acre_lot'].isnull().map({True: 'No acre_lot', False: 'Has acre_lot'}),
            y=np.log10(df['price']),
            ax=ax5
        )
        ax5.set_xlabel('Acre Lot Information')
        ax5.set_ylabel('Log(Price)')
        ax5.set_title('Price Distribution by Acre Lot Availability')
        st.pyplot(fig5)
    
    with col2:
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        sns.boxplot(
            x=df['acre_lot'].isnull().map({True: 'No acre_lot', False: 'Has acre_lot'}),
            y=np.log10(df['house_size']),
            ax=ax6
        )
        ax6.set_xlabel('Acre Lot Information')
        ax6.set_ylabel('Log(House Size)')
        ax6.set_title('House Size Distribution by Acre Lot')
        st.pyplot(fig6)
    
    st.success("""
    **Conclusion:**
    - Properties WITHOUT acre_lot data are smaller (likely condos/apartments)
    - Price distributions are similar (both urban and suburban properties)
    - **This is MNAR** - missingness relates to property type (urban apartments typically don't report lot size)
    """)
    
    st.markdown("---")
    
    # Summary
    st.header("📝 Missingness Summary")
    
    st.markdown("""
    | Feature | Missingness Type | Reason | Strategy |
    |---------|-----------------|--------|----------|
    | `prev_sold_date` | **MNAR** | New properties never sold | Create `has_prev_sale` binary feature |
    | `house_size` | **MAR** | Correlated with bed/bath, depends on status | Regression imputation |
    | `bed` | **MAR** | Correlated with house_size/bath | Regression imputation |
    | `bath` | **MAR** | Correlated with house_size/bed | Regression imputation |
    | `acre_lot` | **MNAR** | Urban apartments/condos don't report | Impute with state median |
    | `zip_code`, `city`, `state`, `price` | **MCAR** | Random errors | Drop rows |
    """)

st.markdown("---")
st.markdown("### 👉 Next: See **Data Preprocessing & Handling** in the next page")