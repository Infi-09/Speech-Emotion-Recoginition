# Speech Emotion Recoginition

* The idea behind creating this project was to build a machine learning model that could detect emotions from the speech we have with each other all the time. Nowadays personalization is something that is needed in all the things we experience everyday. 

* So why not have a emotion detector that will guage your emotions and in the future recommend you different things based on your mood. 
This can be used by multiple industries to offer different services like marketing company suggesting you to buy products based on your emotions, automotive industry can detect the persons emotions and adjust the speed of autonomous cars as required to avoid any collisions etc.

## Datasets:
Kaggle is a platform that caters to data science enthusiasts and it contains a collection of datasets, including RAVDESS, CREMA, TESS, and SAVEE. These datasets are specifically designed for speech emotion recognition applications and are commonly used in research and development of SER systems.
Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) 
Crowd Sourced Emotional Multimodal Actors Dataset (CREMA)
Toronto emotional speech set (TESS)
Speech Emotion Annotated data for emotion recognition systems(SAVEE)


## Model and ffmpeg
### ffmpeg
Download ffmpeg and move ffmped folder to the projcet folder. [ffmpeg](https://ffmpeg.org/download.html)

### Model
The Pretrained Model is uploaded in the drive. [Model](https://drive.google.com/file/d/1UlJOSSe83QpxeicjsZOK0G3g0yKQKolG/view?usp=share_link)

## Audio files:
Tested out the audio files by plotting out the waveform and a spectrogram to see the sample audio files.<br>
**Waveform**
<br>
![](images/wave.png?raw=true)
<br>
<br>
**Spectrogram**<br>
![](images/spec.png?raw=true)
<br>

These are array of values with lables appended to them. 

## Building Models

Since the project is a classification problem, **Convolution Neural Network** seems the obivious choice. We also built **Multilayer perceptrons** and **Long Short Term Memory** models but they under-performed with very low accuracies which couldn't pass the test while predicting the right emotions.

Building and tuning a model is a very time consuming process. The idea is to always start small without adding too many layers just for the sake of making it complex. After testing out with layers, the model which gave the max validation accuracy against test data was little more than 70%
<br>
<br>
![](images/model.png?raw=true)
<br>

## Conclusion
Building the model was a challenging task as it involved lot of trail and error methods, tuning etc. The model is very well trained to distinguish between male and female voices and it distinguishes with 100% accuracy. The model was tuned to detect emotions with more than 70% accuracy. Accuracy can be increased by including more audio files for training.
