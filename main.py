import os
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
import kagglehub
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

print("Path to dataset files:", path)

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
   
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc
data_path = path
import os
print("Dataset path:", data_path)
print("Sample files:", os.listdir(data_path)[:10])
features = []
labels = []
target_emotions = {
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '01': 'neutral'
}

print("Starting Feature Extraction...")
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_parts = file.split('-')
            emotion_code = file_parts[2] 
            
            if emotion_code in target_emotions:
                file_path = os.path.join(root, file)
                mfcc_data = extract_mfcc(file_path)
                features.append(mfcc_data)
                labels.append(target_emotions[emotion_code])

print(f"Processed {len(features)} audio files.")
X = np.array(features)
y = np.array(labels)
le = LabelEncoder()
y = to_categorical(le.fit_transform(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

print(f"Train shape: {X_train.shape}")

model = Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(40, 1)))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax')) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test))
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()