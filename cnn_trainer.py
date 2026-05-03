import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CSV_PATH = "features.csv"

try:
    features_df = pd.read_csv(CSV_PATH)
    print("Data file loaded successfully")
except FileNotFoundError:
    print(f"The file at path {CSV_PATH} not found")
    exit()


X = features_df.drop('genre_label',axis=1)
y = features_df['genre_label']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print("printing shapes and verifying")

# print(f"the shape of X_train_scaled is {X_train_scaled.shape}")
# print(f"the shape of X_test_scaled is {X_test_scaled.shape}")


print("Reshaping data for cnn model")

X_train_cnn = np.expand_dims(X_train_scaled,axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled,axis=-1)

print("\nShapes after reshaping for CNN:")
print(f"X_train_cnn shape: {X_train_cnn.shape}") 
print(f"X_test_cnn shape: {X_test_cnn.shape}")   

print(f"\ny_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")