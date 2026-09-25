import numpy as np
import pandas as pd


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add the great-circle restaurant-to-delivery distance in kilometers using the notebook's Haversine formula."""
    df = df.copy()
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [
            df["Restaurant_latitude"],
            df["Restaurant_longitude"],
            df["Delivery_location_latitude"],
            df["Delivery_location_longitude"],
        ],
    )
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine_a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    central_angle = 2 * np.arctan2(
        np.sqrt(haversine_a), np.sqrt(1 - haversine_a)
    )
    df["Distance_km"] = radius * central_angle
    return df


def drop_location_and_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove raw location coordinates and identifiers after their information is captured by Distance_km."""
    df = df.copy()
    columns_to_drop = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "ID",
        "Delivery_person_ID",
    ]
    return df.drop(columns=columns_to_drop, errors="ignore")


def add_is_weekend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add an Is_Weekend flag for Saturday or Sunday after parsing Order_Date with the notebook's date format."""
    df = df.copy()
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"], format="%d-%m-%Y", errors="coerce"
    )
    df["Is_Weekend"] = df["Order_Date"].dt.dayofweek.isin([5, 6]).astype(int)
    return df


def add_prep_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add restaurant preparation minutes, correcting midnight rollover with 1440 minutes and mean-imputing missing durations."""
    df = df.copy()
    df["Time_Orderd"] = pd.to_datetime(
        df["Time_Orderd"], format="%H:%M:%S", errors="coerce"
    )
    df["Time_Order_picked"] = pd.to_datetime(
        df["Time_Order_picked"], format="%H:%M:%S", errors="coerce"
    )
    df["Prep_Time_min"] = (
        df["Time_Order_picked"] - df["Time_Orderd"]
    ).dt.total_seconds() / 60
    df.loc[df["Prep_Time_min"] < 0, "Prep_Time_min"] += 1440
    mean_prep = df["Prep_Time_min"].mean()
    df["Prep_Time_min"] = df["Prep_Time_min"].fillna(mean_prep)
    return df


def drop_time_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove raw order and pickup timestamps after Prep_Time_min captures their useful duration information."""
    df = df.copy()
    return df.drop(columns=["Time_Orderd", "Time_Order_picked"], errors="ignore")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the notebook's engineered features in order and remove its redundant source columns."""
    df = df.copy()
    df = add_distance_feature(df)
    df = drop_location_and_identifier_columns(df)
    df = add_is_weekend_feature(df)
    df = add_prep_time_feature(df)
    df = drop_time_source_columns(df)
    return df
