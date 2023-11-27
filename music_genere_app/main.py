from flask import Flask, render_template, request, flash, redirect
import librosa
import numpy as np
import pickle

app = Flask(__name__)

# Load the Logistic Regression model using pickle
with open('./random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Function to extract features from uploaded audio file
def extract_features(file):
    y, sr = librosa.load(file, duration=30)  # Load 30 seconds of the audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempogram = librosa.feature.tempogram(y=y, sr=sr)

    features = np.hstack((mfccs.mean(axis=1), spectral_centroid.mean(), tempogram.mean(axis=1)))
    return features.reshape(1, -1)  # Return features as a 2D array


# Route to handle file upload and genre prediction
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, browser submits an empty file without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            # Call function to extract features from uploaded audio file
            features = extract_features(file)
            features = np.array(features)

            genres = ["Bollypop", "Carnatic", "Ghazal", "Semiclassical", "Sufi"]
            # Use the loaded Logistic Regression model to predict the genre using extracted features
            # genre = model.predict(features)[0]  # Assuming model.predict returns an array, selecting the first element
            genre = genres[model.predict(features)[0]]
            # Return the predicted genre to display
            #return render_template('result.html', genre=genre)
            return render_template('index.html', genre=genre)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
