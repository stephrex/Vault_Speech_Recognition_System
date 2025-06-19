import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow_keras as tf
from sklearn.preprocessing import StandardScaler
import io
import soundfile as sf

# Load your trained model (.keras)
model = tf.keras.models.load_model(
    "CNN_Model.keras")

# Custom preprocessing function (per-sample scaling)


def preprocess_audio_file(file, target_sr=16000, n_mfcc=40, fixed_length=90):
    x, sr = librosa.load(file, sr=target_sr)

    # Pad or trim to at least 1 second of audio
    if len(x) < target_sr:
        x = np.pad(x, (0, target_sr - len(x)), 'constant')
    else:
        x = x[:target_sr]

    # Compute MFCC features
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.moveaxis(mfccs, 1, 0)  # shape: (time, features)

    # Pad or truncate to fixed length (time axis)
    if mfccs.shape[0] < fixed_length:
        pad_width = fixed_length - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:fixed_length, :]

    # Per-sample scaling
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs)

    # Reshape for CNN input (batch, time, features, 1)
    mfccs_scaled = mfccs_scaled.reshape(
        1, mfccs_scaled.shape[0], mfccs_scaled.shape[1], 1)

    return mfccs_scaled


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
        prediction = model.predict(processed_audio)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        st.success(
            f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
