import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
CSV_PATH = 'features.csv'

try:
    features_df = pd.read_csv(CSV_PATH)
    # print("DataFrame info")
    # print(features_df.info())
    # print('\nStatistical Summary')
    # print(features_df.describe())
    # print(features_df.isnull().sum())
    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]
    # sns.set_style("whitegrid")
    # plt.figure(figsize=(12,6))

    # ax = sns.countplot(x='genre_label',data=features_df,palette='viridis')
    # ax.set_title("Distribution of music genres in dataset")
    # ax.set_xlabel('Genre', fontsize=12)
    # ax.set_ylabel('Number of Segments', fontsize=12)

    
    # ax.set_xticklabels(genre_names, rotation=30)

    
    # plt.tight_layout()
    # plt.show()
     
    # print("genrating box plot for spectral centorid")
    # plt.figure(figsize=(14,7))

    # box_ax = sns.boxplot(x="genre_label",y="25",data=features_df,palette='cubehelix')
    # box_ax.set_title("Spectral centroid for distribution accros genres")
    # box_ax.set_xlabel("genres",fontsize=14)
    # box_ax.set_ylabel("spectral centroid",fontsize=14)
    # box_ax.set_xticklabels(genre_names,rotation=30,ha='right')

    # plt.tight_layout()
    # plt.show()

    # print("Genrating the violin plot for the first MFCC ")

    # plt.figure(figsize=(14,7))
    # violin_ax = sns.violinplot(x='genre_label',y='0',data=features_df,palette='Spectral')
    # violin_ax.set_title("First MFCC Distribution across genres",fontsize=18)
    # violin_ax.set_xlabel("Genre")
    # violin_ax.set_ylabel("MFCC 1 value",fontsize=14)
    # violin_ax.set_xticklabels(genre_names,rotation=30,ha='right')

    # plt.tight_layout()
    # plt.show()
    
    # print("Cpmputing the correlation matrix")
    # corr_mat = features_df.corr()
    # print("Correlation matrix computed successfully")
    # print("Top 5 rows of the correlation matrix")
    # print(corr_mat.head())

    # print("Heatmap of correaltion matix")
    # plt.figure(figsize=(18,15))
    # sns.heatmap(corr_mat,cmap='coolwarm',annot=False)
    # plt.title("heatmap of correaltion matrix",fontsize=20)
    # plt.tight_layout()
    # plt.show()/

    X = features_df.drop('genre_label',axis=1)
    y = features_df['genre_label']

    


    # print(f"Shape of X is {X.shape}")
    # print(f"Shape of y is {y.shape}")

    # print(X.head())
    # print("-----Y------")
    # print(y.head())/
     
    # print("Label encoding step")
    # if np.issubdtype(y.dtype,np.integer):
    #     print("Already all are neumerical")
    # else:
    #     print("Labels are not neumerically encoded")        
    
    # print("Splitting Dataset into Training and Testing Data")

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)

    # print("Verification through shapes")
    # print(f"Shape of X_train: {X_train.shape}")
    # print(f"Shape of X_test: {X_test.shape}")   
    # print(f"Shape of y_train: {y_train.shape}") 
    # print(f"Shape of y_test: {y_test.shape}")   

    print("scaling features")
    scaler = StandardScaler()
   

    # print("Verification of scaler")
    # print(scaler.mean_[:5])
    # print(scaler.scale_[:5])
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # print("verification of scaled data")

    # print(f"mean of first 5 xtrain scaled data {X_train_scaled[:,:5].mean(axis=0)}")
    # print(f"std of first 5 xtrain scaled data {X_train_scaled[:,:5].std(axis=0)}")
    
    # print(f"Mean of first 5 scaled test data {X_test_scaled[:,:5].mean(axis=0)}")

    print("Training the logistic regression model")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled,y_train)
    print("Logistic regression model trained Successfully")
    print(f"Model learned the following classes {log_reg.classes_}")
except FileNotFoundError:
    print(f"Error: The file '{CSV_PATH}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")