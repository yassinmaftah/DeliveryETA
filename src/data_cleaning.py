import pandas as pd
from pathlib import Path

Project_Forlder = Path(__file__).resolve().parents[1]
# print(Project_Forlder)
df = pd.read_csv(Project_Forlder / 'data' / 'raw' / 'dataset.csv')

def ana():
    print(f"sheap :{df.shape}")
    print("#" * 50)
    print(f"Columns :{df.columns.tolist()}")
    print("#" * 50)
    print(df.info())
    print("#" * 50)
    print(df.describe())
    print("#" * 50)
    print(df.head(10))
    print("#" * 50)
    print(df.isnull().sum())
    print("#" * 50)
    print(df[df.isnull().any(axis=1)].shape[0], "rows have at least one missing value")
    print("#" * 50)

    for col in ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']:
        print(col, ":", df[col].unique())
        print("-" * 30)
        
    print("Duplicate rows:", df.duplicated().sum())

    missing_exp = df[df['Courier_Experience_yrs'].isnull()]
    not_missing_exp = df[df['Courier_Experience_yrs'].notnull()]

    print("Missing exp - avg Delivery_Time_min:", missing_exp['Delivery_Time_min'].mean())
    print("Not missing - avg Delivery_Time_min:", not_missing_exp['Delivery_Time_min'].mean())
    print("-" * 30)
    print("Missing exp - Vehicle_Type distribution:")
    print(missing_exp['Vehicle_Type'].value_counts(normalize=True))
    print("Not missing - Vehicle_Type distribution:")
    print(not_missing_exp['Vehicle_Type'].value_counts(normalize=True))
    

def load_data(path: Path) -> pd.DataFrame :
    return pd.read_csv(path)


def hundel_mission_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    categories_columns = ['Weather', 'Traffic_Level', 'Time_of_Day']
    
    for cat in categories_columns :
        mode_value = df[cat].mode()[0]
        df[cat] = df[cat].fillna(mode_value)
        
    median_exp = df['Courier_Experience_yrs'].median()
    df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(median_exp)
    
    return df

