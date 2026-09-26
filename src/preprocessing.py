from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ['Delivery_person_Age', 'Delivery_person_Ratings',
                     'Distance_km', 'Prep_Time_min', 'multiple_deliveries',
                     'Road_traffic_density']

CATEGORICAL_FEATURES = ['Weatherconditions', 'Type_of_order', 'Type_of_vehicle', 'City']

BINARY_FEATURES = ['Festival', 'Is_Weekend']


def build_preprocessor():
    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # for the mission values 
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", one_hot_encoder),
    ])

    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("bin", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop", # neeed documentation
    )
    return preprocessor