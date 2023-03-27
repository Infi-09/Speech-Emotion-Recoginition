import random, os
from keyword_spotting_service import Keyword_Spotting_Service
from flask import Flask, request, jsonify, render_template

modelPath = 'models/modelV2.'

app = Flask(__name__)

@app.route('/')
def landing():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # Get audio file and save it
    audioFile = request.files.get("file")
    fileName = str(random.randint(0, 100000))
    audioFile.save(fileName)

    # Create a Instance for Keyword Spotting Service
    kss = Keyword_Spotting_Service()

    # Make a predition
    preds = kss.predict(fileName)

    # Remove the audio file, we don't need it anymore
    os.remove(fileName)

    # Send back the prediction in json format
    data = {"Prediction": preds}

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
