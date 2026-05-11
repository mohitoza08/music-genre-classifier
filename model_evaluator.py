import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

    print("Genrating Predictions")
    y_pred_log_reg = log_reg_model.predict(X_test_scaled)
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)

    y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
    y_pred_cnn = np.argmax(y_pred_cnn_probs,axis=1)

    print("Predictions Genarated")

    print("Verifying Shapes")
    print(f"Logistic Regression Predictions Shape: {y_pred_log_reg.shape}")
    print(f"SVM Predictions Shape: {y_pred_svm.shape}")
    print(f"Random Forest Predictions Shape: {y_pred_rf.shape}")
    print(f"CNN Predictions Shape: {y_pred_cnn.shape}")


    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]

    # print("Classification Reports")

    # print("Classification Report of Logistic Regression")
    # print(classification_report(y_test,y_pred_log_reg,target_names=genre_names))
    
    # print("Classification Report of SVM")
    # print(classification_report(y_test,y_pred_svm,target_names=genre_names))
    
    # print("Classification Report of RandomForest")
    # print(classification_report(y_test,y_pred_rf,target_names=genre_names))
    
    # print("Classification Report of CNN Model")
    # print(classification_report(y_test,y_pred_cnn,target_names=genre_names))
    
    print("Confusion Matrixs")
    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)

    # print("Confusion Matrix of Logistic Regression")
    # print(cm_log_reg)
    
    # print("Confusion Matrix of SVM")
    # print(cm_svm)
    
    # print("Confusion Matrix of Randomforest")
    # print(cm_rf)
    
    # print("Confusion Matrix of Cnn")
    # print(cm_cnn)
    

    def plot_confusion_matrix(cm,label,title,ax):
        sns.heatmap(    
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label,
            yticklabels=label,
            ax=ax
        )
        ax.set_title(title,fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True label')

    fig,axes  = plt.subplots(2,2,figsize=(15,12))
    fig.suptitle('Confusion Matrix for all model')
    plot_confusion_matrix(cm_log_reg, genre_names, 'Logistic Regression', axes[0, 0])
    plot_confusion_matrix(cm_svm, genre_names, 'Support Vector Machine', axes[0, 1])
    plot_confusion_matrix(cm_rf, genre_names, 'Random Forest', axes[1, 0])
    plot_confusion_matrix(cm_cnn, genre_names, 'Convolutional Neural Network', axes[1, 1])

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
except FileNotFoundError as e:
    print(f"Error File {e.filename} Not found")
except Exception as e:
    print(f"Unexpected Error occured {e}")



