import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
CSV_PATH = 'features.csv'

try:
    features_df = pd.read_csv(CSV_PATH)
    print("DataFrame info")
    print(features_df.info())
    print('\nStatistical Summary')
    print(features_df.describe())
    print(features_df.isnull().sum())
    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]
    sns.set_style("whitegrid")
    plt.figure(figsize=(12,6))

    ax = sns.countplot(x='genre_label',data=features_df,palette='viridis')
    ax.set_title("Distribution of music genres in dataset")
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Number of Segments', fontsize=12)

    
    ax.set_xticklabels(genre_names, rotation=30)

    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{CSV_PATH}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")