import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

st.header("Enter Order Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Delivery Person Age", min_value=15, max_value=60, value=29)
with col2:
    rating = st.slider("Delivery Person Rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)

col1, col2 = st.columns(2)
with col1:
    distance = st.number_input("Distance (km)", min_value=1.0, max_value=80.0, value=10.0)
with col2:
    prep_time = st.number_input("Preparation Time (minutes)", min_value=5, max_value=30, value=10)

col1, col2 = st.columns(2)
with col1:
    multiple_deliveries = st.selectbox("Multiple Deliveries at Once", options=[0, 1, 2, 3])
with col2:
    traffic = st.selectbox("Road Traffic Density", options=["Low", "Medium", "High", "Jam"])

col1, col2 = st.columns(2)
with col1:
    festival = st.selectbox("Is it a Festival Day?", options=["No", "Yes"])
with col2:
    weekend = st.selectbox("Is it a Weekend?", options=["No", "Yes"])

col1, col2 = st.columns(2)
with col1:
    weather = st.selectbox("Weather Conditions", options=["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy"])
with col2:
    order_type = st.selectbox("Type of Order", options=["Snack", "Meal", "Drinks", "Buffet"])

col1, col2 = st.columns(2)
with col1:
    vehicle_type = st.selectbox("Type of Vehicle", options=["motorcycle", "scooter", "electric_scooter", "bicycle"])
with col2:
    city_type = st.selectbox("City Type", options=["Metropolitian", "Urban", "Semi-Urban"])



Project_Folder = Path(__file__).resolve().parents[1]
model = joblib.load(Project_Folder / 'models' / 'model.pkl')
preprocessor = joblib.load(Project_Folder / 'models' / 'scaler.pkl')

if st.button("Predict Delivery Time"):
    input_data = pd.DataFrame([{
        'Delivery_person_Age': age,
        'Delivery_person_Ratings': rating,
        'Distance_km': distance,
        'Prep_Time_min': prep_time,
        'multiple_deliveries': multiple_deliveries,
        'Road_traffic_density': {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}[traffic],
        'Festival': 1 if festival == 'Yes' else 0,
        'Is_Weekend': 1 if weekend == 'Yes' else 0,
        'Weatherconditions': weather,
        'Type_of_order': order_type,
        'Type_of_vehicle': vehicle_type,
        'City': city_type,
    }])

    input_processed = preprocessor.transform(input_data)

    prediction = model.predict(input_processed)[0]

    st.success(f"Estimated Delivery Time: {prediction:.1f} minutes")
    
    
    
@st.cache_data
def load_and_split_data():
    df = pd.read_csv(Project_Folder / 'data' / 'processed' / 'cleaned_dataset.csv')
    y = df['Time_taken(min)']
    X = df.drop(columns=['Time_taken(min)'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_and_split_data()

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

y_train_pred = model.predict(X_train_processed)
y_test_pred = model.predict(X_test_processed)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

st.header("Model Performance")

st.subheader("Train vs Test — Overfitting Check")
col1, col2 = st.columns(2)
with col1:
    st.write("**Training Set**")
    st.metric("MAE", f"{train_mae:.2f} min")
    st.metric("RMSE", f"{train_rmse:.2f} min")
    st.metric("R²", f"{train_r2:.3f}")
with col2:
    st.write("**Test Set**")
    st.metric("MAE", f"{test_mae:.2f} min")
    st.metric("RMSE", f"{test_rmse:.2f} min")
    st.metric("R²", f"{test_r2:.3f}")