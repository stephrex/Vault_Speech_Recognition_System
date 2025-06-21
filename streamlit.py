import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tf_keras as tf
from sklearn.preprocessing import StandardScaler
import io
import soundfile as sf
from pydub import AudioSegment

# Load model
model = tf.models.load_model("CNN2_Model.keras")

class_map = {
    0: "Close Vault",
    1: "Open Vault",
    2: "Unrecognized"
}

def convert_audio_to_wav_16k_mono(uploaded_file_bytes, format_hint):
    audio = AudioSegment.from_file(io.BytesIO(
        uploaded_file_bytes), format=format_hint)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Export to in-memory WAV file
    out_io = io.BytesIO()
    audio.export(out_io, format="wav")
    out_io.seek(0)
    return out_io

# Preprocess audio function


def preprocess_audio_file(file_like, target_sr=16000, n_mfcc=40, fixed_length=90):
    x, sr = librosa.load(file_like, sr=target_sr)

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

    return mfccs_scaled.reshape(1, mfccs_scaled.shape[0], mfccs_scaled.shape[1], 1)


# Streamlit UI
st.title("🎤 Audio Classifier (Custom Test)")

uploaded_file = st.file_uploader(
    "Upload an audio file (Supported formats: wav, mp3, opus, ogg, flac, m4a)",
    type=["wav", "mp3", "opus", "ogg", "flac", "m4a"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # Use filename to guess format if needed
    file_format = uploaded_file.name.split('.')[-1]

    try:
        st.info("📦 Converting audio to WAV mono 16kHz...")
        wav_io = convert_audio_to_wav_16k_mono(
            file_bytes, format_hint=file_format)

        st.audio(wav_io, format='audio/wav')

        # Visualize waveform
        st.write("Waveform:")
        wav_io.seek(0)
        audio_np, sr = sf.read(wav_io)
        fig, ax = plt.subplots()
        librosa.display.waveshow(audio_np, sr=sr, ax=ax)
        st.pyplot(fig)

        # Rewind before prediction
        wav_io.seek(0)
        processed_audio = preprocess_audio_file(wav_io)
        prediction = model.predict(processed_audio)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        class_label = class_map.get(predicted_class, "Unknown")

        st.success(
            f"Predicted Class: {class_label} (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"Error during processing: {e}")
