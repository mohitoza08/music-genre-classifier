import streamlit as st

st.title("Music Genre Classifier")

st.write("Upload the file and our model will predict its genre")

uploaded_file  = st.file_uploader("Choose a file",type="wav")

if uploaded_file is not None:
    predicted_genre="Rock"
    st.success(f"the predicted genre is :{predicted_genre}")