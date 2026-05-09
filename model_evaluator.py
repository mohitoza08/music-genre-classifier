import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split


print("Now Model evalutaion")

try:
    print("Preparing Test Data")

    features_df = pd.read_csv('features.csv')

    X= features_df.drop('genre_label',axis=1)
    y= features_df['genre_label']

    _,X_test,_,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

    print(f"The shape of X Test is {X_test.shape}")
    print(f"The shape of Y test is {y_test.shape}")
     
    print("Loading scaler and other models")

    scaler  = joblib.load('scaler.joblib')

    log_reg_model =  joblib.load('logistic_regression_model.joblib')
    svm_model = joblib.load('Svm_model.joblib')
    rf_model = joblib.load('random_forest_model.joblib')

    print("Models Loaded Successfully")
    print(f"Scaler: {type(scaler)}")
    print(f"Logistic Regression Model: {type(log_reg_model)}")
    print(f"SVM Model: {type(svm_model)}")
    print(f"Random Forest Model: {type(rf_model)}")


    print("Loading Cnn model")

    cnn_model = tf.keras.models.load_model('music_genre_cnn.h5')

    print("Cnn model Loaded Successfully")
    print(f"Cnn model {type(cnn_model)}")
 

    print("preparing Data for evaluation")
    X_test_scaled = scaler.transform(X_test)

    print("Shape extending of test data")
    X_test_cnn =  np.expand_dims(X_test_scaled,axis=-1)
    print(f"Shape of X_test_cnn (for Keras): {X_test_cnn.shape}")


    print("All models and data loaded Successfully")


except FileNotFoundError as e:
    print(f"Error File {e.filename} Not found")
except Exception as e:
    print(f"Unexpected Error occured {e}")



