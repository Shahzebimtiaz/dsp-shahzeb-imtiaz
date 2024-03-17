import pandas as pd
import joblib
from house_prices.preprocess import preprocess_data, feature_engineering

def make_predictions(df: pd.DataFrame) -> pd.DataFrame:
    # Load model
    model = joblib.load('../models/model.joblib')
    
    # Preprocess and feature engineering
    X_processed = preprocess_data(df, categorical_columns, numerical_columns)
    X_featured = feature_engineering(X_processed, categorical_columns, numerical_columns)
    
    # Make predictions
    predictions = model.predict(X_featured)
    
    return predictions
