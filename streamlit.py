import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tf_keras as tf
from sklearn.preprocessing import StandardScaler
import io
import soundfile as sf

# Load your trained model (.keras)
model = tf.models.load_model("CNN4_Model.keras")

# Class label mapping
class_map = {
    0: "Unrecognized",
    1: "Open Vault",
    2: "Close Vault"
}


def pad_audio(x, sr, max_duration=4):
    max_len = sr * max_duration
    if len(x) < max_len:
        return np.pad(x, (0, max_len - len(x)), mode='constant')
    else:
        return x[:max_len]


def preprocess_audio_file(file, target_sr=16000, n_mfcc=40, fixed_length=90, ndim=3):

    x, sr = librosa.load(file, sr=target_sr)

    x = pad_audio(x, sr)

    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.moveaxis(mfccs, 1, 0)

    if mfccs.shape[0] < fixed_length:
        pad_width = fixed_length - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:fixed_length, :]

    if ndim == 3:
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
    elif ndim == 2:
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1])
    else:
        raise ValueError("Invalid ndim. Must be 2 or 3.")

    return mfccs


# Streamlit App UI
st.title("🎤 Audio Classifier (Custom Test)")

uploaded_file = st.file_uploader(
    "Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Read and decode audio
    audio_bytes = uploaded_file.read()
    audio_np, sr = sf.read(io.BytesIO(audio_bytes))

    st.audio(uploaded_file, format='audio/wav')

    # Plot waveform
    st.write("Waveform:")
    fig, ax = plt.subplots()
    librosa.display.waveshow(audio_np, sr=sr, ax=ax)
    st.pyplot(fig)

    # Save to temp file for librosa
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(audio_bytes)

    # Preprocess & Predict
    try:
        processed_audio = preprocess_audio_file(temp_file_path)
        print(processed_audio)
        print(processed_audio.shape)
        st.write(processed_audio)
        prediction_prob = model.predict(processed_audio)
        st.write(prediction_prob)
        prediction_prob = np.squeeze(prediction_prob)
        st.write(prediction_prob)
        predicted_index = np.argmax(prediction_prob)
        st.write(predicted_index)
        predicted_class = np.argmax(prediction_prob)
        st.write(predicted_class)
        confidence = prediction_prob[predicted_class]

        # Map predicted class to label
        class_label = class_map.get(predicted_class, "Unknown")
        st.write(class_label)

        st.success(
            f"Predicted Class: {class_label} (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
