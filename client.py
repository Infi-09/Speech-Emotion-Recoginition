import json
import requests

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH = "ravdess/Actor_24/03-01-01-01-01-01-24.wav"

if __name__ == "__main__":

    audioFile = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audioFile, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted Keyword: {data['Predictions']}")
