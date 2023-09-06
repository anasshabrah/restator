import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import IncrementalPCA, PCA
import tensorflow as tf

categorical_cols = ['State', 'City', 'Distric', 'Number of Rooms', 'Age of Building', 
                    'Number of Bathrooms', 'Balcony', 'Furnished', 'Within a Building Complex', 
                    'Title Deed Status', 'Floor Number', 'Number of Floors']

def load_data(path='data.csv'):
    return pd.read_csv(path)

def preprocess_data(df):
    df = pd.get_dummies(df, columns=categorical_cols)
    X = df.drop(columns=['Price'])
    y = df['Price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def normalize_and_impute(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_imputer = SimpleImputer(strategy="mean")
    y_train = y_imputer.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = y_imputer.transform(y_test.values.reshape(-1, 1)).flatten()

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_test = imputer.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test

def create_and_train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train, X_test, y_train, y_test = normalize_and_impute(X_train, X_test, y_train, y_test)

    model = create_and_train_model(X_train, y_train)

    y_pred = model.predict(X_test).flatten()

    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
