import os
from sklearn.preprocessing import StandardScaler
import tf_keras as tf
import numpy as np
import librosa
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = 'CNN2_Model.keras'
model = tf.models.load_model(MODEL_PATH)


def preprocess_audio(file_path):
    target_sr = 16000
    n_mfcc = 40
    fixed_length = 90

    # Load audio file
    x, sr = librosa.load(file_path, sr=target_sr)

    # Pad if too short
    if len(x) < target_sr:
        x = np.pad(x, (0, target_sr - len(x)), mode='constant')

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.moveaxis(mfcc, 1, 0)  # (Time, Features)

    # Ensure fixed length
    if mfcc.shape[0] < fixed_length:
        pad_width = fixed_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:fixed_length, :]

    # Per-sample scaling
    scaler = StandardScaler()
    # Fit and transform this single sample
    mfcc_scaled = scaler.fit_transform(mfcc)

    # Reshape for CNN: (1, time, features, channel)
    mfcc_scaled = mfcc_scaled.reshape(
        1, mfcc_scaled.shape[0], mfcc_scaled.shape[1], 1)

    return mfcc_scaled


@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    audio_file = None

    if request.method == 'POST':
        file = request.files['audio']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            features = preprocess_audio(file_path)
            pred = model.predict(features)
            predicted_class = np.argmax(pred, axis=1)[0]
            prediction = f"Predicted Class: {predicted_class}"

            audio_file = file.filename

    return render_template("index.html", prediction=prediction, audio_file=audio_file)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
