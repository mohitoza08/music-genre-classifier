import os
import json
import librosa
import numpy as np
import pandas as pd

sample_rate = 22050
track_duration_seconds = 30
num_segments = 10

NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

samples_per_track = sample_rate * track_duration_seconds

Dataset_path = "genres_original"
json_path = "data.json"
CSV_PATH = "features.csv"   

def process_dataset(Dataset_path, csv_path):

    data = {
        "mapping": [],
        "labels": [],
        "features": []
    }

    print("Starting feature extraction...")

    for i, genre_folder in enumerate(sorted(os.listdir(Dataset_path))):
        genre_path = os.path.join(Dataset_path, genre_folder)

        if os.path.isdir(genre_path):
            print(f"\nProcessing genre: {genre_folder}\n")
            data["mapping"].append(genre_folder)
           
            for filename in sorted(os.listdir(genre_path)):
                if filename.endswith('.wav'):
                    file_path = os.path.join(genre_path,filename)

                    try:
                        signal, sr = librosa.load(file_path,sr=sample_rate)
                        if len(signal)>=samples_per_track:
                            num_sample_per_segment = int(samples_per_track/num_segments)

                            for s in range(num_segments):
                                start_sample = s * num_sample_per_segment
                                end_sample = start_sample + num_sample_per_segment
                                segment = signal[start_sample:end_sample]

                                mfccs = librosa.feature.mfcc(y=segment,
                                                             sr=sr,
                                                             n_mfcc=NUM_MFCC,
                                                             n_fft = N_FFT,
                                                             hop_length = HOP_LENGTH)
                                mfccs_processed =  np.mean(mfccs,axis=1)
                                chroma = librosa.feature.chroma_stft(y=segment,
                                                                        sr=sr,
                                                                        n_fft = N_FFT,
                                                                        hop_length = HOP_LENGTH)    
                                chroma_processed = np.mean(chroma, axis=1)
                                spectral_centroid = librosa.feature.spectral_centroid(y=segment,
                                                                                      sr=sr,
                                                                                      n_fft=N_FFT,
                                                                                      hop_length=HOP_LENGTH)
                                spectral_centroid_processed = np.mean(spectral_centroid)
                                spectral_rolloff = librosa.feature.spectral_rolloff(y=segment,
                                                                                    sr=sr,
                                                                                    n_fft = N_FFT,
                                                                                    hop_length=HOP_LENGTH)
                                spectral_rolloff_processed = np.mean(spectral_rolloff)
                                zcr = librosa.feature.zero_crossing_rate(y=segment,
                                                                         hop_length=HOP_LENGTH)
                                zcr_processed  = np.mean(zcr)
                                feature_vector = np.hstack((mfccs_processed, 
                                                             chroma_processed, 
                                                             spectral_centroid_processed, 
                                                             spectral_rolloff_processed, 
                                                             zcr_processed))
                                data["features"].append(feature_vector.tolist())
                                data["labels"].append(i)
                                print(f"Processed segment {s+1}/{num_segments} of {filename}")
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        continue    

    print("\nConverting data to pandas DataFrame...")
    features_df = pd.DataFrame(data["features"])
    features_df["genre_label"] = data["labels"]
    
    # --- THIS IS THE FINAL, NEW CODE BLOCK ---
    # Save the DataFrame to a CSV file
    print(f"Saving DataFrame to {csv_path}...")
    features_df.to_csv(csv_path, index=False)
    print("Feature extraction complete..")


if __name__ == "__main__":
    process_dataset(Dataset_path, CSV_PATH)