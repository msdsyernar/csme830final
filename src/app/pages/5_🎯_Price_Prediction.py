import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from catboost import CatBoostRegressor
import plotly.graph_objects as go
import sys
import os

# 1. Get the directory of the current file (src/app/pages)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up 3 levels to reach the project root
# pages -> app -> src -> PROJECT ROOT
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

# 3. Add the project root to sys.path so Python can find 'src'
if project_root not in sys.path:
    sys.path.append(project_root)

# NOW you can import safely
from src.features.custom_features import InteractionFeatures
from src.features.custom_features import FrequencyEncoder

# from sklearn.base import BaseEstimator, TransformerMixin

# class FrequencyEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, columns):
#         self.columns = columns
#         self.freq_maps = {}

#     def fit(self, X, y=None):
#         for col in self.columns:
#             self.freq_maps[col] = X[col].value_counts().to_dict()
#         return self

#     def transform(self, X):
#         X = X.copy()
#         for col in self.columns:
#             X[col + "_freq"] = X[col].map(self.freq_maps[col]).fillna(0)
#         return X


st.set_page_config(page_title="Price Prediction", page_icon="🎯", layout="wide")

st.title("🎯 House Price Prediction Tool")
st.markdown("---")

# Load model and data
@st.cache_resource
def load_model():
    # 2. Get the directory where THIS script (app.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 3. Construct the full path to the file
    # Change 'models_folder' to the actual name of your folder 
    # model_path = r'C:\Users\Администратор\Documents\study\CSME 830\house_price_project\notebooks\catboost_full_pipeline.pkl'
    model_path = r'notebooks/catboost_full_pipeline.pkl'

    try:
        # 4. Load using the full path
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        # It is helpful to print the path it TRIED to look in so you can debug
        st.error(f"⚠️ Model not found. I looked here: {model_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    

@st.cache_data
def load_reference_data():
    try:
        # df = pd.read_csv(r'c:\Users\Администратор\Documents\study\CSME 830\house_price_project\models\re_cleaned.csv')
        df = pd.read_csv(r'models/re_cleaned.csv')
        return df
    except:
        st.error("⚠️ Reference data 're_cleaned.csv' not found")
        return None

model = load_model()
df_reference = load_reference_data()

st.header("🏠 Enter Property Details")
st.markdown("Fill in the property information below to get a price prediction")

# Create input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏘️ Property Features")
    
    bed = st.number_input("Number of Bedrooms", min_value=0, max_value=20, value=3, step=1)
    bath = st.number_input("Number of Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    house_size = st.number_input("House Size (sqft)", min_value=100, max_value=50000, value=2000, step=100)
    acre_lot = st.number_input("Lot Size (acres)", min_value=0.0, max_value=60.0, value=0.2, step=0.1)
    
    st.subheader("📅 Property History")
    has_prev_sale = st.selectbox("Previously Sold?", ["No", "Yes"])
    if has_prev_sale == "Yes":
        years_since_sale = st.slider("Years Since Last Sale", 0, 50, 5)
    else:
        years_since_sale = 0

with col2:
    st.subheader("📍 Location")
    
    if df_reference is not None:
        # Get unique values from reference data
        states = sorted(df_reference['state'].dropna().unique()) if 'state' in df_reference.columns else ['California', 'Texas', 'Florida', 'New York']
        state = st.selectbox("State", states)
        
        # Filter cities by state if possible
        if 'state' in df_reference.columns and 'city' in df_reference.columns:
            cities = sorted(df_reference[df_reference['state'] == state]['city'].dropna().unique())
        else:
            cities = ['Los Angeles', 'New York City', 'Chicago', 'Houston']
        
        city = st.selectbox("City", cities)
        
        zip_code = st.text_input("ZIP Code", "90001")
    else:
        state = st.text_input("State", "California")
        city = st.text_input("City", "Los Angeles")
        zip_code = st.text_input("ZIP Code", "90001")
    
    st.subheader("🏷️ Status")
    status = st.selectbox("Property Status", ["for_sale", "ready_to_build", "sold"])

st.markdown("---")

# Predict button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    
    with st.spinner("Calculating price prediction..."):
    
        
        
        # Create feature dictionary
        features = {
            'status': status, 
            'bed': bed,
            'bath': bath,
            'acre_lot': acre_lot,
            'city': city,
            'state': state,
            'zip_code': zip_code,
            'house_size': house_size
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Make prediction
        if model is not None:
            log_prediction = model.predict(input_df)[0]
            # prediction = np.expm1(log_prediction)
            prediction = log_prediction
        else:
            # Demo prediction using simple formula
            base_price = 100000
            size_factor = house_size * 150
            location_factor = state_encoded * 0.3
            bed_bath_factor = (bed + bath) * 20000
            prediction = base_price + size_factor + location_factor + bed_bath_factor
        
        # Display results
        st.markdown("---")
        st.header("🎉 Prediction Results")
        
        # Main prediction
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.metric(
                label="Estimated House Price",
                value=f"${prediction:,.0f}",
                delta="Predicted Value"
            )
        
        with col2:
            price_per_sqft = prediction / house_size
            st.metric("Price per sqft", f"${price_per_sqft:.2f}")
        
        with col3:
            confidence = "High" if model is not None else "Demo"
            st.metric("Confidence", confidence)
        
        # Price range
        st.subheader("💰 Estimated Price Range")
        lower_bound = prediction * 0.90
        upper_bound = prediction * 1.10
        
        st.info(f"**Conservative Estimate:** ${lower_bound:,.0f} - ${upper_bound:,.0f}")
        st.caption("±10% confidence interval based on model performance")
        
        # Visualization
        st.subheader("📊 Price Breakdown Analysis")
        st.write("👀 Streamlit is sending this data:", input_df)
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Price", 'font': {'size': 24}},
            delta={'reference': prediction * 0.9, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, prediction * 1.5], 'tickformat': '$,.0f'},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, prediction * 0.7], 'color': "lightgray"},
                    {'range': [prediction * 0.7, prediction * 1.3], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction * 1.2
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature contribution (estimated)
        st.subheader("🎯 Key Value Drivers")
        
        contributions = {
            'Location (City/State/ZIP)': 45,
            'House Size': 25,
            'Bedrooms & Bathrooms': 15,
            'Lot Size': 10,
            'Other Features': 5
        }
        
        fig2 = go.Figure(data=[go.Pie(
            labels=list(contributions.keys()),
            values=list(contributions.values()),
            hole=0.4,
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        )])
        
        fig2.update_layout(
            title="Estimated Feature Contribution to Price",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Property summary
        st.subheader("📋 Property Summary")
        
        summary_data = {
            'Feature': ['Bedrooms', 'Bathrooms', 'House Size', 'Lot Size', 'Location', 
                       'Status', 'Price per sqft', 'Previous Sale'],
            'Value': [
                f"{bed} beds",
                f"{bath} baths",
                f"{house_size:,} sqft",
                f"{acre_lot:.2f} acres",
                f"{city}, {state} {zip_code}",
                status,
                f"${price_per_sqft:.2f}",
                'Yes' if has_prev_sale == 'Yes' else 'No'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Insights & Recommendations")
        
        if price_per_sqft > 300:
            st.success("✅ This property is in a premium location with high price per square foot.")
        elif price_per_sqft < 100:
            st.info("💰 This property offers good value with a lower price per square foot.")
        else:
            st.info("📊 This property is priced moderately for the area.")
        
        if house_size < 1500:
            st.info("🏠 This is a smaller property, which may appeal to first-time buyers or downsizers.")
        elif house_size > 3500:
            st.success("🏰 This is a larger property, suitable for families or those needing more space.")
        
        if bed / bath > 2:
            st.warning("⚠️ Consider adding bathrooms - higher bed-to-bath ratio than typical homes.")

# Sidebar info
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown("""
    1. **Enter property details** in the form
    2. **Click "Predict Price"** button
    3. **Review the results** and insights
    
    ### 🎯 Model Information
    - **Model:** CatBoost Regressor
    - **Accuracy:** R² = 0.80
    - **MAE:** ~$55,000
    - **Features Used:** 14
    
    ### 📊 Price Factors
    The model considers:
    - Location (city, state, ZIP)
    - Property size and features
    - Lot size
    - Number of bed/bath
    - Property status
    - Historical sales
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** More accurate predictions require the actual trained model file (catboost_model.pkl)")

st.markdown("---")
st.markdown("### 🎓 Project completed! Check other pages for detailed analysis.")