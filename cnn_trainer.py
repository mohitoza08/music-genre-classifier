import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)

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


# print("Reshaping data for cnn model")

X_train_cnn = np.expand_dims(X_train_scaled,axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled,axis=-1)

# print("\nShapes after reshaping for CNN:")
# print(f"X_train_cnn shape: {X_train_cnn.shape}") 
# print(f"X_test_cnn shape: {X_test_cnn.shape}")   

# print(f"\ny_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")



model = Sequential()
model.add(Conv1D(
      filters=32,
      kernel_size=3,
      activation='relu',
      input_shape=(X_train_cnn.shape[1],1)
))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())


model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(units=64,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(units=10,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled Successfully, Now ready to be trained....")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("Traing the CNN model")
history = model.fit(
    X_train_cnn,
    y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)
print("Model Training Completed")

model.summary()


import matplotlib.pyplot as plt

def plot_history(history):
    fig,axs = plt.subplots(2,1,figsize=(10,10))
    axs[0].plot(history.history['accuracy'],label="Training Accuracy")
    axs[0].plot(history.history['val_accuracy'],label="Validation Acuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Training vs Validattion Accuracy")
    axs[0].legend(loc='lower right')


    axs[1].plot(history.history['loss'],label="Training loss")
    axs[1].plot(history.history['val_loss'],label="Validation loss")
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].legend(loc='upper right')

    plt.tight_layout()

    plt.show()

plot_history(history)    

model.save("music_genre_cnn.h5")
print("Model successfully saved")