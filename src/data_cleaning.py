from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw delivery dataset from the supplied CSV path."""
    return pd.read_csv(path)


def fix_numeric_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert disguised numeric text to numeric values after removing whitespace and literal NaN markers."""
    df = df.copy()
    column_fix = ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]

    for column in column_fix:
        df[column] = df[column].str.strip()
        df[column] = df[column].replace("NaN", np.nan)
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def impute_delivery_person_age(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ages with the mean because the notebook found the age distribution's skew close to zero."""
    df = df.copy()
    mean_age = df["Delivery_person_Age"].mean()
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(mean_age)
    return df


def impute_delivery_person_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ratings with the median because the notebook observed strong negative skew."""
    df = df.copy()
    median_rating = df["Delivery_person_Ratings"].median()
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(median_rating)
    return df


def impute_multiple_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing multiple-delivery counts with the most frequent value."""
    df = df.copy()
    mode_deliveries = df["multiple_deliveries"].mode()[0]
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(mode_deliveries)
    return df


def cast_cleaned_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast age and multiple-delivery counts to integers after their missing values are filled."""
    df = df.copy()
    df["Delivery_person_Age"] = df["Delivery_person_Age"].astype(int)
    df["multiple_deliveries"] = df["multiple_deliveries"].astype(int)
    return df


def clean_weatherconditions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the conditions prefix, trim whitespace, convert the nan marker, and fill missing weather with its mode."""
    df = df.copy()
    df["Weatherconditions"] = df["Weatherconditions"].str.replace(
        "conditions", "", regex=False
    ).str.strip()
    df["Weatherconditions"] = df["Weatherconditions"].replace("nan", np.nan)
    mode_weather = df["Weatherconditions"].mode()[0]
    df["Weatherconditions"] = df["Weatherconditions"].fillna(mode_weather)
    return df


def clean_time_taken(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the minute value after the (min) prefix and convert the delivery-time target to numeric."""
    df = df.copy()
    df["Time_taken(min)"] = df["Time_taken(min)"].str.split(" ").str[1]
    df["Time_taken(min)"] = pd.to_numeric(df["Time_taken(min)"], errors="coerce")
    return df


def clean_city(df: pd.DataFrame) -> pd.DataFrame:
    """Trim city labels and convert their literal NaN marker to a missing value for later imputation."""
    df = df.copy()
    df["City"] = df["City"].str.strip()
    df["City"] = df["City"].replace("NaN", np.nan)
    return df


def clean_type_of_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    """Trim surrounding whitespace from vehicle categories without changing their labels."""
    df = df.copy()
    df["Type_of_vehicle"] = df["Type_of_vehicle"].str.strip()
    return df


def impute_city(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing city values with the most frequent city, matching the notebook's mode-based choice."""
    df = df.copy()
    mode_city = df["City"].mode()[0]
    df["City"] = df["City"].fillna(mode_city)
    return df


def fix_gps_sign_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Use absolute restaurant coordinates because the notebook identified flipped signs as GPS errors."""
    df = df.copy()
    df["Restaurant_latitude"] = df["Restaurant_latitude"].abs()
    df["Restaurant_longitude"] = df["Restaurant_longitude"].abs()
    return df


def drop_near_zero_gps_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any restaurant or delivery coordinate whose absolute value is below 1."""
    df = df.copy()
    zero_mask = (
        (df["Restaurant_latitude"].abs() < 1)
        | (df["Restaurant_longitude"].abs() < 1)
        | (df["Delivery_location_latitude"].abs() < 1)
        | (df["Delivery_location_longitude"].abs() < 1)
    )
    return df[~zero_mask]


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Check for exact duplicate rows while preserving every row because the notebook found none to remove."""
    df = df.copy()
    _duplicate_count = int(df.duplicated().sum())
    return df


def clean_traffic_and_festival(df: pd.DataFrame) -> pd.DataFrame:
    """Trim traffic-density and festival labels and convert their literal NaN markers to missing values."""
    df = df.copy()
    for column in ["Road_traffic_density", "Festival"]:
        df[column] = df[column].str.strip()
        df[column] = df[column].replace("NaN", np.nan)
    return df


def impute_traffic_and_festival(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing traffic-density and festival values with their respective modes."""
    df = df.copy()
    mode_traffic = df["Road_traffic_density"].mode()[0]
    df["Road_traffic_density"] = df["Road_traffic_density"].fillna(mode_traffic)
    mode_festival = df["Festival"].mode()[0]
    df["Festival"] = df["Festival"].fillna(mode_festival)
    return df


def drop_temp_city_code(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the temporary city-code column when it is present in the intermediate data."""
    df = df.copy()
    if "temp_city_code" in df.columns:
        df = df.drop(columns=["temp_city_code"])
    return df


def check_iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the notebook's 1.5-IQR bounds for delivery time, age, and ratings without removing rows."""
    df = df.copy()
    for column in ["Time_taken(min)", "Delivery_person_Age", "Delivery_person_Ratings"]:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower) | (df[column] > upper)]
        _outlier_count = len(outliers)
    return df


def fix_invalid_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Treat ratings above 5 as missing and fill them with the median, matching the notebook's invalid-value decision."""
    df = df.copy()
    df.loc[df["Delivery_person_Ratings"] > 5, "Delivery_person_Ratings"] = np.nan
    median_rating = df["Delivery_person_Ratings"].median()
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(median_rating)
    return df


def fix_invalid_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ages of 15 and 50 with missing, then fill them with the recalculated mean and cast to integer."""
    df = df.copy()
    df.loc[df["Delivery_person_Age"] == 50, "Delivery_person_Age"] = np.nan
    df.loc[df["Delivery_person_Age"] == 15, "Delivery_person_Age"] = np.nan
    mean_age = df["Delivery_person_Age"].mean()
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(mean_age).astype(int)
    return df


def keep_time_taken_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Recheck the delivery-time IQR upper bound and keep flagged rows because the notebook treated them as legitimate delays."""
    df = df.copy()
    q1 = df["Time_taken(min)"].quantile(0.25)
    q3 = df["Time_taken(min)"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    outliers = df[df["Time_taken(min)"] > upper_bound]
    _outlier_count = len(outliers)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's Step 1 cleaning operations in execution order and return a new DataFrame."""
    df = df.copy()
    df = fix_numeric_column_dtypes(df)
    df = impute_delivery_person_age(df)
    df = impute_delivery_person_ratings(df)
    df = impute_multiple_deliveries(df)
    df = cast_cleaned_numeric_columns(df)
    df = clean_weatherconditions(df)
    df = clean_time_taken(df)
    df = clean_city(df)
    df = clean_type_of_vehicle(df)
    df = impute_city(df)
    df = fix_gps_sign_errors(df)
    df = drop_near_zero_gps_rows(df)
    df = check_duplicates(df)
    df = clean_traffic_and_festival(df)
    df = impute_traffic_and_festival(df)
    df = drop_temp_city_code(df)
    df = check_iqr_outliers(df)
    df = fix_invalid_ratings(df)
    df = fix_invalid_ages(df)
    df = keep_time_taken_outliers(df)
    return df
