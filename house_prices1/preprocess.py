import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df: pd.DataFrame, categorical_columns: list[str], numerical_columns: list[str]) -> pd.DataFrame:
    df_copy = df.copy()
    
    # Impute missing numerical values with mean
    numerical_imputer = SimpleImputer(strategy='mean')
    df_copy[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
    
    # Impute missing categorical values with a placeholder
    df_copy[categorical_columns] = df_copy[categorical_columns].fillna('Unknown')
    
    return df_copy

def feature_engineering(df: pd.DataFrame, categorical_columns: list[str], numerical_columns: list[str]) -> pd.DataFrame:
    # One-hot encode categorical variables and scale numerical variables
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ], remainder='passthrough')
    
    transformed_data = preprocessor.fit_transform(df)
    
    return transformed_data
