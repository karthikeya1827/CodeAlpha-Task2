# 🎙️ Emotion Recognition from Speech using CNN

This project builds a deep learning model to recognize human emotions (happy, sad, angry, neutral) from speech audio using the [RAVDESS Emotional Speech Audio Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).

## 📌 Objective
To classify emotions from speech signals using Mel-Frequency Cepstral Coefficients (MFCCs) and a Convolutional Neural Network (CNN).

---

## 🧠 Approach
- **Feature Extraction**: MFCCs from `.wav` files
- **Preprocessing**: Label encoding, one-hot encoding, train-test split
- **Model Architecture**: CNN with Conv1D, MaxPooling, Dropout, and Dense layers
- **Evaluation**: Accuracy on test data and training history visualization

---

## 📁 Dataset
- **Source**: [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Emotions Used**: Happy (`03`), Sad (`04`), Angry (`05`), Neutral (`01`)
- **Audio Format**: `.wav`

---

## 🧪 Features
- MFCCs (40 coefficients averaged over time)
- Emotion labels encoded and one-hot transformed
- CNN model trained for 50 epochs with batch size 64

---

## 📊 Results
- **Test Accuracy**: ~`XX.XX%` *(replace with your actual result)*
- **Training Visualization**:
  - Accuracy vs Epochs
  - Validation Accuracy vs Epochs

---

## 🧰 Libraries Used
- `librosa`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- `sklearn` for preprocessing
- `tensorflow.keras` for model building
- `kagglehub` for dataset download

---

## 🚀 How to Run
1. Clone the repo
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
