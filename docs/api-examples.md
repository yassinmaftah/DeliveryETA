# DeliveryETA — Programmatic Inference & API Reference

This document provides example inputs, Python integration code, and expected outputs for programmatic inference using the trained DeliveryETA model artifacts (`models/model.pkl` and `models/scaler.pkl`).

---

## 1. Input Schema & Data Types

The preprocessor expects a `pandas.DataFrame` containing the following 12 features before transformation:

| Feature Name | Type | Description | Valid Values / Example |
|---|---|---|---|
| `Delivery_person_Age` | `int` / `float` | Courier age in years | `18` – `60` |
| `Delivery_person_Ratings` | `float` | Courier rating (1 to 5) | `1.0` – `5.0` |
| `Distance_km` | `float` | Great-circle distance between restaurant and customer | `0.1` – `50.0+` |
| `Prep_Time_min` | `float` | Restaurant food preparation time | `5.0` – `45.0` |
| `multiple_deliveries` | `int` | Number of simultaneous orders assigned | `0`, `1`, `2`, `3` |
| `Road_traffic_density` | `int` | Ordinal traffic level (0: Low, 1: Medium, 2: High, 3: Jam) | `0`, `1`, `2`, `3` |
| `Festival` | `int` | Binary flag for festival days (0: No, 1: Yes) | `0`, `1` |
| `Is_Weekend` | `int` | Binary flag for weekend orders (0: Weekday, 1: Weekend) | `0`, `1` |
| `Weatherconditions` | `str` | Dominant meteorological condition | `"Sunny"`, `"Cloudy"`, `"Fog"`, `"Stormy"`, `"Sandstorms"`, `"Windy"` |
| `Type_of_order` | `str` | Meal category | `"Snack"`, `"Meal"`, `"Drinks"`, `"Buffet"` |
| `Type_of_vehicle` | `str` | Vehicle operated by courier | `"motorcycle"`, `"scooter"`, `"electric_scooter"`, `"bicycle"` |
| `City` | `str` | Urban development tier | `"Metropolitian"`, `"Urban"`, `"Semi-Urban"` |

---

## 2. Python Inference Snippet

```python
import joblib
import pandas as pd
from pathlib import Path

# 1. Load serialized model and preprocessor
PROJECT_ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(PROJECT_ROOT / "models" / "model.pkl")
preprocessor = joblib.load(PROJECT_ROOT / "models" / "scaler.pkl")

# 2. Construct raw input payload
orders = pd.DataFrame([
    {
        "Delivery_person_Age": 29,
        "Delivery_person_Ratings": 4.8,
        "Distance_km": 6.4,
        "Prep_Time_min": 10.0,
        "multiple_deliveries": 0,
        "Road_traffic_density": 0,  # Low
        "Festival": 0,
        "Is_Weekend": 0,
        "Weatherconditions": "Sunny",
        "Type_of_order": "Meal",
        "Type_of_vehicle": "scooter",
        "City": "Metropolitian",
    },
    {
        "Delivery_person_Age": 38,
        "Delivery_person_Ratings": 4.1,
        "Distance_km": 18.2,
        "Prep_Time_min": 25.0,
        "multiple_deliveries": 2,
        "Road_traffic_density": 3,  # Jam
        "Festival": 1,
        "Is_Weekend": 1,
        "Weatherconditions": "Stormy",
        "Type_of_order": "Buffet",
        "Type_of_vehicle": "motorcycle",
        "City": "Urban",
    }
])

# 3. Transform inputs using the fitted ColumnTransformer
processed_features = preprocessor.transform(orders)

# 4. Generate ETA predictions (in minutes)
predictions = model.predict(processed_features)

for i, eta in enumerate(predictions):
    print(f"Order #{i + 1} Estimated Delivery Time: {eta:.1f} minutes")
```

---

## 3. Sample Request / Response Test Cases

### Test Case A: Optimal Fast Delivery (Short distance, Sunny, Low traffic)
```json
{
  "Delivery_person_Age": 28,
  "Delivery_person_Ratings": 4.9,
  "Distance_km": 2.5,
  "Prep_Time_min": 8.0,
  "multiple_deliveries": 0,
  "Road_traffic_density": 0,
  "Festival": 0,
  "Is_Weekend": 0,
  "Weatherconditions": "Sunny",
  "Type_of_order": "Snack",
  "Type_of_vehicle": "bicycle",
  "City": "Metropolitian"
}
```
**Expected Prediction:** `~16.5 minutes`

---

### Test Case B: Adverse Conditions (Long distance, Stormy, Traffic Jam, Festival)
```json
{
  "Delivery_person_Age": 42,
  "Delivery_person_Ratings": 3.8,
  "Distance_km": 21.0,
  "Prep_Time_min": 20.0,
  "multiple_deliveries": 2,
  "Road_traffic_density": 3,
  "Festival": 1,
  "Is_Weekend": 1,
  "Weatherconditions": "Stormy",
  "Type_of_order": "Meal",
  "Type_of_vehicle": "motorcycle",
  "City": "Urban"
}
```
**Expected Prediction:** `~46.8 minutes`
