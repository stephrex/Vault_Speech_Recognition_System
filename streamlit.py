import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tf_keras as tf
from sklearn.preprocessing import StandardScaler
import io
import soundfile as sf
from audiorecorder import audiorecorder  # pip install streamlit-audiorec

# Load model
model = tf.models.load_model("CNN2_Model.keras")

class_map = {
    0: "Close Vault",
    1: "Open Vault",
    2: "Unrecognized"
}

def preprocess_audio_file(file, target_sr=16000, n_mfcc=40, fixed_length=90):
    x, sr = librosa.load(file, sr=target_sr)

    if len(x) < target_sr:
        x = np.pad(x, (0, target_sr - len(x)), 'constant')
    else:
        x = x[:target_sr]

    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.moveaxis(mfccs, 1, 0)

    if mfccs.shape[0] < fixed_length:
        pad_width = fixed_length - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:fixed_length, :]

    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs)
    mfccs_scaled = mfccs_scaled.reshape(1, mfccs_scaled.shape[0], mfccs_scaled.shape[1], 1)

    return mfccs_scaled

st.title("🎤 Audio Classifier with Live Recording")

st.write("## Record audio:")
audio = audiorecorder("Click to record", "Recording...")

if len(audio) > 0:
    st.audio(audio.tobytes(), format="audio/wav")

    with open("recorded.wav", "wb") as f:
        f.write(audio.tobytes())

    try:
        processed_audio = preprocess_audio_file("recorded.wav")
        prediction = model.predict(processed_audio)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        class_label = class_map.get(predicted_class, "Unknown")

        st.success(f"Predicted: {class_label} (Confidence: {confidence:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

st.write("## Or upload an audio file:")
uploaded_file = st.file_uploader("Upload (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    audio_np, sr = sf.read(io.BytesIO(audio_bytes))
    st.audio(uploaded_file, format='audio/wav')

    fig, ax = plt.subplots()
    librosa.display.waveshow(audio_np, sr=sr, ax=ax)
    st.pyplot(fig)

    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)

    try:
        processed_audio = preprocess_audio_file("temp_audio.wav")
        prediction = model.predict(processed_audio)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        class_label = class_map.get(predicted_class, "Unknown")

        st.success(f"Predicted: {class_label} (Confidence: {confidence:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
