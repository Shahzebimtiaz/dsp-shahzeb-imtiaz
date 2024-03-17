import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from house_prices.preprocess import preprocess_data, feature_engineering

def build_model(df: pd.DataFrame, target_column: str, categorical_columns: list[str], numerical_columns: list[str]) -> dict[str, float]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Preprocess and feature engineering
    X_processed = preprocess_data(X, categorical_columns, numerical_columns)
    X_featured = feature_engineering(X_processed, categorical_columns, numerical_columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_featured, y, test_size=0.25, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, '../models/model.joblib')
    
    # Evaluate model
    rmse = model.score(X_test, y_test)
    
    return {'rmse': rmse}
