import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tf_keras as tf
import io

# Load model
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


def preprocess_audio_file(file_bytes, target_sr=16000, n_mfcc=40, fixed_length=90, ndim=3):
    # Read with librosa from file-like object
    audio_stream = io.BytesIO(file_bytes)
    x, sr = librosa.load(audio_stream, sr=target_sr, mono=True)

    # Pad or trim
    x = pad_audio(x, sr)

    # Extract MFCC
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.moveaxis(mfccs, 1, 0)

    if mfccs.shape[0] < fixed_length:
        pad_width = fixed_length - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:fixed_length, :]

    if ndim == 3:
        mfccs = mfccs.reshape(1, fixed_length, n_mfcc, 1)
    elif ndim == 2:
        mfccs = mfccs.reshape(1, fixed_length, n_mfcc)
    else:
        raise ValueError("Invalid ndim. Must be 2 or 3.")

    return mfccs


# Streamlit UI
st.title("🎤 Audio Classifier (Custom Test)")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    file_bytes = uploaded_file.read()

    # Display waveform for confirmation
    x_disp, sr_disp = librosa.load(io.BytesIO(file_bytes), sr=None)
    st.write("Waveform:")
    fig, ax = plt.subplots()
    librosa.display.waveshow(x_disp, sr=sr_disp, ax=ax)
    st.pyplot(fig)

    try:
        # Preprocess and predict
        processed_audio = preprocess_audio_file(file_bytes)
        prediction_prob = model.predict(processed_audio)
        prediction_prob = np.squeeze(prediction_prob)

        predicted_index = int(np.argmax(prediction_prob))
        confidence = float(prediction_prob[predicted_index])
        class_label = class_map.get(predicted_index, "Unknown")

        st.success(
            f"Predicted Class: {class_label} (Confidence: {confidence:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
