import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import joblib



def main():
    st.title("Music Genre Classification App")

    st.write("Welcome This Application uses Random Forest To"
             " Predict The Genre of Music track")
    
    uploaded_file = st.file_uploader("Drag and Drop your file",type=['wav'])
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' Uploaded Successfully!")

    def laod_model_and_scaler():
        try:
            model = joblib.load('random_forest_model.joblib',compile=False)
            scaler = joblib.load('scaler.joblib')  
            genre_mapping = {
            0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
            5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
        }
        except FileNotFoundError as e:
            st.error(f"error loading model or file")
            st.stop()
        except  Exception as e:
            st.error("Unexpected error Occured while loading model or scaler")
            st.stop()         
    def extract_features(audio_file,sample_rate=22050,  n_mfcc=13,n_chroma=12):
        try:
            y,sr = librosa.load(audio_file,sr=sample_rate,duration=30)
            
            #Extract Features

            #mfcc 
            mfcc = librosa.load.feature.mfcc(y=y,sr=sr,n_mfcc=n_chroma)
            mfcc_mean = np.mean(mfcc,axis=1)

            #chroma 
            chroma = librosa.load.feature.chroma_stft(y=y,sr=sr,n_chroma=n_chroma)
            chroma_mean = np.mean(chroma, axis=1)
        
            # Spectral Centroid
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_cent_mean = np.mean(spec_cent)
            
            # Spectral Rolloff
            spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spec_roll_mean = np.mean(spec_roll)
            
            # Zero-Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            
            # Concatenate all features into a single vector
            features = np.concatenate([
                mfcc_mean,
                chroma_mean,
                np.array([spec_cent_mean]),
                np.array([spec_roll_mean]),
                np.array([zcr_mean])
            ])
            
            return features

        except Exception as e:
            st.error(f"Error processing audio file: {e}")
            return None
if __name__ == '__main__':
    main()
