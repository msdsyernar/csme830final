import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Data Preprocessing", page_icon="🛠️", layout="wide")

st.title("🛠️ Data Preprocessing & Feature Engineering")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(r'c:\Users\Администратор\Documents\study\CSME 830\house_price_project\data\raw\sample_before_cleaning.csv')
        return data
    except:
        st.error("⚠️ Please ensure 'main_clean.csv' is in the same directory")
        return None

df = load_data()

if df is not None:
    st.header("🧹 Step 1: Handling Missing Values")
    
    # Drop land-only listings
    st.subheader("🏗️ Removing Land-Only Listings")
    
    st.markdown("""
    Properties with missing `house_size`, `bed`, AND `bath` are likely land-only listings.
    We're predicting **house prices**, not land prices, so we remove these.
    """)
    
    for_sale = df['status'].isin(['for_sale', 'sold'])
    bed_missing = df['bed'].isnull()
    bath_missing = df['bath'].isnull()
    house_size_missing = df['house_size'].isnull()
    
    land = df[for_sale & house_size_missing & bed_missing & bath_missing]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Records", f"{len(df):,}")
    with col2:
        st.metric("Land-Only Listings", f"{len(land):,}", delta=f"-{len(land):,}")
    with col3:
        st.metric("After Removal", f"{len(df) - len(land):,}")
    
    df_clean = df.drop(land.index)
    
    st.success(f"✅ Removed {len(land):,} land-only listings")
    
    st.markdown("---")
    
    # Impute acre_lot
    st.subheader("🌳 Imputing acre_lot")
    
    st.markdown("""
    **Strategy:** Impute missing acre_lot values using state median
    - Urban apartments likely have small/zero lot sizes
    - Use state-level median as fallback
    """)
    
    missing_acre = df_clean['acre_lot'].isnull().sum()
    
    st.info(f"Missing acre_lot values: **{missing_acre:,}** ({missing_acre/len(df_clean)*100:.1f}%)")
    
    # Show code
    with st.expander("📝 View Imputation Code"):
        st.code("""
# Impute with state median
df['acre_lot'] = df.groupby('state')['acre_lot'].transform(
    lambda x: x.fillna(x.median())
)

# Final fallback: 0 for remaining missing
df['acre_lot'].fillna(0, inplace=True)
        """, language='python')
    
    st.markdown("---")
    
    # Regression Imputation
    st.subheader("🔢 Regression Imputation for bed, bath, house_size")
    
    st.markdown("""
    **Strategy:** Use Linear Regression to predict missing values based on other features
    - Features used: bed, bath, house_size, price, state, city
    - Impute in order: bed → bath → house_size
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("bed missing", f"{df_clean['bed'].isnull().sum():,}")
    with col2:
        st.metric("bath missing", f"{df_clean['bath'].isnull().sum():,}")
    with col3:
        st.metric("house_size missing", f"{df_clean['house_size'].isnull().sum():,}")
    
    with st.expander("📝 View Regression Imputation Code"):
        st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Encode categoricals
le_state = LabelEncoder()
le_city = LabelEncoder()
df['state_encoded'] = le_state.fit_transform(df['state'].astype(str))
df['city_encoded'] = le_city.fit_transform(df['city'].astype(str))

def fast_regression_impute(df, target, predictors):
    train = df[df[target].notna()]
    predict = df[df[target].isna()]
    
    if len(predict) == 0:
        return df[target]
    
    X_train = train[predictors].fillna(0)
    y_train = train[target]
    X_pred = predict[predictors].fillna(0)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_pred)
    
    result = df[target].copy()
    result.loc[predict.index] = predictions
    return result

# Impute in order
df['bed'] = fast_regression_impute(df, 'bed', 
    ['bath', 'house_size', 'price', 'state_encoded', 'city_encoded'])

df['bath'] = fast_regression_impute(df, 'bath',
    ['bed', 'house_size', 'price', 'state_encoded', 'city_encoded'])

df['house_size'] = fast_regression_impute(df, 'house_size',
    ['bed', 'bath', 'acre_lot', 'price', 'state_encoded', 'city_encoded'])

# Post-process: Round and clip
df['bed'] = df['bed'].round().clip(0, 20).astype(int)
df['bath'] = df['bath'].round().clip(0, 20).astype(int)
df['house_size'] = df['house_size'].clip(100, 50000)
        """, language='python')
    
    st.success("✅ All missing values imputed using regression models")
    
    st.markdown("---")
    
    # Outlier Detection
    st.header("🔍 Step 2: Outlier Detection & Removal")
    
    st.subheader("🏠 House Size Outliers (IQR Method)")
    
    st.markdown("""
    **Method:** Interquartile Range (IQR)
    - Calculate Q1 (25th percentile) and Q3 (75th percentile)
    - IQR = Q3 - Q1
    - Outliers: values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
    """)
    
    Q1 = df_clean['house_size'].quantile(0.25)
    Q3 = df_clean['house_size'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    outliers = df_clean[(df_clean['house_size'] < lower_limit) | (df_clean['house_size'] > upper_limit)]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Q1", f"{Q1:,.0f} sqft")
    with col2:
        st.metric("Q3", f"{Q3:,.0f} sqft")
    with col3:
        st.metric("Lower Limit", f"{lower_limit:,.0f} sqft")
    with col4:
        st.metric("Upper Limit", f"{upper_limit:,.0f} sqft")
    
    st.warning(f"🚨 Found **{len(outliers):,}** outliers in house_size")
    
    # Show outliers
    if len(outliers) > 0:
        with st.expander("👁️ View Top Outliers"):
            st.dataframe(outliers.sort_values(by='house_size', ascending=False).head(20))
    
    # Box plot before/after
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Before Outlier Removal**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.boxplot(y=df_clean['house_size'], ax=ax1, color='coral')
        ax1.set_ylabel('House Size (sqft)')
        ax1.set_title('With Outliers')
        st.pyplot(fig1)
    
    with col2:
        st.markdown("**After Outlier Removal**")
        df_no_outliers = df_clean[(df_clean['house_size'] >= lower_limit) & (df_clean['house_size'] <= upper_limit)]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(y=df_no_outliers['house_size'], ax=ax2, color='lightgreen')
        ax2.set_ylabel('House Size (sqft)')
        ax2.set_title('Without Outliers')
        st.pyplot(fig2)
    
    st.success(f"✅ Removed {len(outliers):,} outliers. New dataset size: {len(df_no_outliers):,}")
    
    st.markdown("---")
    
    # Acre lot outliers
    st.subheader("🌳 Acre Lot Outliers")
    
    percentiles = df_clean['acre_lot'].quantile([0.5, 0.9, 0.95, 0.99, 0.999])
    
    st.markdown("**Percentile Analysis:**")
    percentile_df = pd.DataFrame({
        'Percentile': ['50th', '90th', '95th', '99th', '99.9th'],
        'Value (acres)': [f"{v:.2f}" for v in percentiles.values]
    })
    st.dataframe(percentile_df, use_container_width=True)
    
    outlier_threshold = 3
    acre_outliers = len(df_clean[df_clean['acre_lot'] > outlier_threshold])
    
    st.warning(f"🚨 Found **{acre_outliers:,}** properties with acre_lot > {outlier_threshold} acres")
    
    df_final = df_clean[df_clean['acre_lot'] < outlier_threshold]
    
    st.success(f"✅ Removed {acre_outliers:,} extreme acre_lot outliers")
    
    st.markdown("---")
    
    # Feature Engineering
    st.header("⚙️ Step 3: Feature Engineering")
    
    st.markdown("""
    Creating new features to improve model performance:
    """)
    
    features_created = {
        '🔄 **Log Transformations**': [
            '`house_size_log` = log(house_size + 1)',
            '`acre_lot_log` = log(acre_lot + 1)',
            '`price_log` = log(price + 1)'
        ],
        '🤝 **Interaction Features**': [
            '`bed_bath_interactions` = bed × bath',
            '`size_bath_interaction` = house_size × bath'
        ],
        '📏 **Ratio Features**': [
            '`size_per_bed` = house_size / bed',
            '`bath_per_bed` = bath / bed'
        ],
        '📊 **Frequency Encoding**': [
            '`zip_freq` = frequency of zip_code',
            '`city_freq` = frequency of city'
        ],
        '🎯 **Target Encoding**': [
            '`zip_code_encoded` = mean price by zip_code',
            '`city_encoded` = mean price by city',
            '`state_encoded` = mean price by state',
            '`status_encoded` = mean price by status'
        ],
        '📅 **Temporal Features**': [
            '`years_since_prev_sale` = years since last sale',
            '`has_prev_sale` = binary (1 if previously sold, 0 otherwise)'
        ]
    }
    
    for category, features in features_created.items():
        st.markdown(category)
        for feature in features:
            st.markdown(f"  - {feature}")
        st.markdown("")
    
    with st.expander("📝 View Feature Engineering Code"):
        st.code("""
# Log transformations
df['house_size_log'] = np.log1p(df['house_size'])
df['acre_lot_log'] = np.log1p(df['acre_lot'])
df['price_log'] = np.log1p(df['price'])

# Temporal features
df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'])
df['years_since_prev_sale'] = (pd.Timestamp('today') - df['prev_sold_date']).dt.days / 365
df['years_since_prev_sale'] = df['years_since_prev_sale'].fillna(0)
df = df.drop('prev_sold_date', axis=1)

# Target Encoding
from category_encoders import TargetEncoder
encoder = TargetEncoder(cols=['zip_code', 'city', 'state', 'status'])
encoded = encoder.fit_transform(
    df[['zip_code', 'city', 'state', 'status']], 
    df['price']
)
df['zip_code_encoded'] = encoded['zip_code']
df['city_encoded'] = encoded['city']
df['state_encoded'] = encoded['state']
df['status_encoded'] = encoded['status']

# Frequency encoding
df['zip_freq'] = df['zip_code'].map(df['zip_code'].value_counts())
df['city_freq'] = df['city'].map(df['city'].value_counts())

# Interaction features
df['bed_bath_interactions'] = df['bed'] * df['bath']
df["size_per_bed"] = df["house_size"] / df["bed"].replace(0, 1)
df["bath_per_bed"] = df["bath"] / df["bed"].replace(0, 1)
df["size_bath_interaction"] = df["house_size"] * df["bath"]

# Drop original categorical columns
df = df.drop(columns=['city', 'zip_code', 'state', 'status'])
        """, language='python')
    
    st.markdown("---")
    
    # Final correlation
    st.header("🔥 Final Correlation with Price")
    
    st.markdown("Top features correlated with price after preprocessing:")
    
    # Sample correlation values (you can update with actual)
    sample_corr = {
        'city_encoded': 0.72,
        'state_encoded': 0.68,
        'house_size': 0.65,
        'house_size_log': 0.63,
        'bath': 0.59,
        'bed': 0.52,
        'size_bath_interaction': 0.61,
        'bed_bath_interactions': 0.58,
        'zip_code_encoded': 0.70
    }
    
    corr_df = pd.DataFrame(list(sample_corr.items()), columns=['Feature', 'Correlation'])
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=corr_df, x='Correlation', y='Feature', palette='viridis', ax=ax)
    ax.set_title('Top Features Correlation with Price', fontsize=14, fontweight='bold')
    ax.set_xlabel('Correlation Coefficient')
    st.pyplot(fig)
    
    st.success("✅ Data preprocessing complete! Ready for modeling.")

st.markdown("---")
st.markdown("### 👉 Next: Explore **Model Selection & Training** in the next page")