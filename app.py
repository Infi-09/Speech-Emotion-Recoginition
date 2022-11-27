import pickle
from random import random
import random
import audiofile
import librosa
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from keras.layers import BatchNormalization, LSTM
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense
from keras.layers import Dropout
from flask import Flask, render_template, request, redirect, url_for,jsonify
import tensorflow as tf 

import wave

model = Sequential()
model.add(Conv1D(2048, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(20, 1)))
model.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
model.add(BatchNormalization())

model.add(Conv1D(1024, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
model.add(BatchNormalization())

model.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides = 2, padding = 'same'))
model.add(BatchNormalization())

model.add(LSTM(256, return_sequences=True))

model.add(LSTM(128))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(14, activation='softmax'))

optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

values = {1:'male-calm',
          2:'male-happy',
          3:'male-neutral',
          4:'male-sad'}

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model3.h5")
print("Loaded model from disk")

def predict(audioPath):
  X, sample_rate = audiofile.read(audioPath)
  sample_rate = np.array(sample_rate)

  mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
  featurelive = mfccs
  livedf2 = featurelive

  livedf2 = pd.DataFrame(data=livedf2)
  livedf2 = livedf2[:20]

  live = pd.DataFrame(data=livedf2)
  live = live.stack().to_frame().T

  #livepred = values[random.randint(1, 4)]
  twodim= np.expand_dims(live, axis=2)
  livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)

  encoder = LabelEncoder()
  livepredictions = (encoder.inverse_transform((livepreds)))

  livepredictions = str(livepredictions)
  return livepredictions

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    preds = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            print(file)
            with open('test.weba', 'wb') as f:
                f.write(file.read())

            preds = predict('test.weba')
            print(preds)

    return render_template('index.html', preds=preds)
    
# @app.route("/predict", methods=["POST", "GET"])
# def predict():
#     if request.method == "POST":

#         data = np.array([wheelbase, carlength, curbweight, boreratio, enginesize, horsepower])
#         data = data.reshape(1, 6) 
#         output = model.predict(data)

#         return redirect(url_for("result", price=output))

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    
    if 'audio' in request.files:
        
        file = request.files['audio']
        file.stream.seek(0)
        audioBlob = file.read()
    
        nchannels = 1
        sampwidth = 1
        framerate = 8000
        nframes = 1
        name = 'output.wav'
        audio = wave.open(name, 'wb')
        audio.setnchannels(nchannels)
        audio.setsampwidth(sampwidth)
        audio.setframerate(framerate)
        audio.setnframes(nframes)
        
        audio.writeframes(audioBlob)
    
    return jsonify({'msg': 'success'})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)