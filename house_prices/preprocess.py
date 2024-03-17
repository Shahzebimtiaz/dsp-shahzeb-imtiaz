import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def custom_one_hot_encoder(X):
    # Perform one-hot encoding on categorical features
    encoded_data = []
    for column in X.columns:
        unique_values = X[column].unique()
        encoded_column = np.zeros((len(X), len(unique_values)), dtype=int)
        for i, value in enumerate(unique_values):
            encoded_column[:, i] = (X[column] == value).astype(int)
        encoded_data.append(encoded_column)
    return np.concatenate(encoded_data, axis=1)


def custom_scaler(X_train, X_test):
    # Scale numerical features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def preprocess_data(dataset_raw):
    features = ['MSSubClass', 'LotArea', 'Street', 'LotShape']
    X = dataset_raw[features]
    y = dataset_raw['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    categorical_columns = ['LotShape', 'Street']
    numerical_columns = ['LotArea', 'MSSubClass']

    # Transform numerical features
    numeric_transformer = StandardScaler()
    # Fit the numeric transformer
    numeric_transformer.fit(X_train[numerical_columns])

    # Transform the training data
    X_train_numeric = numeric_transformer.transform(X_train[numerical_columns])

    # Transform the test data
    X_test_numeric = numeric_transformer.transform(X_test[numerical_columns])
    # Perform one-hot encoding on categorical features
    categorical_transformer = custom_one_hot_encoder(
        X_train[categorical_columns]
    )
    X_train_categorical = categorical_transformer
    X_test_categorical = custom_one_hot_encoder(X_test[categorical_columns])

    # Concatenate transformed features
    X_train_final = np.concatenate(
        [X_train_numeric, X_train_categorical],
        axis=1
    )
    X_test_final = np.concatenate([X_test_numeric, X_test_categorical], axis=1)

    return (
            X_test_final,
            X_train_final,
            y_train,
            y_test,
            numeric_transformer,
            categorical_transformer
    )
