import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    print("Cpmputing the correlation matrix")
    corr_mat = features_df.corr()
    print("Correlation matrix computed successfully")
    print("Top 5 rows of the correlation matrix")
    print(corr_mat.head())

    print("Heatmap of correaltion matix")
    plt.figure(figsize=(18,15))
    sns.heatmap(corr_mat,cmap='coolwarm',annot=False)
    plt.title("heatmap of correaltion matrix",fontsize=20)
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError:
    print(f"Error: The file '{CSV_PATH}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")