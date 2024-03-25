# Voice-Driven 3D Facial Emotion Recognition For Mental Health Monitoring

# About Project
This research highlights the significance of understanding users' emotional states to enhance the user experience in recommendation applications, particularly in Point of Interest (POI) recommendations. Previous studies have overlooked emotions and lacked comprehensive datasets. In response, we propose an EmoPOI dataset and a novel approach that integrates facial feature extraction using Convolutional Neural Networks (CNNs) and emotion analysis through Long Short-Term Memory (LSTM) layers. Our method excels in accuracy compared to state-of-the-art techniques, leveraging FER-2013 and EmoPOI datasets.

# Workflow
The first step involves importing essential libraries by installing them on your system. The primary libraries necessary for code execution include:

Python 3.7
Pytorch 1.9.0
CUDA 11.3
Blender 3.4.1
ffmpeg 4.4.1
torch==1.9.0
torchvision==0.10.0
torchaudio==0.9.0
numpy
scipy==1.7.1
librosa==0.8.1
tqdm
pickle
transformers==4.6.1
trimesh==3.9.27
pyrender==0.1.45
opencv-python
   
# Installation
1.	Download the reprository
2.	Create the conda virtual environment using: conda create -n verhm python=3.7
3.	Activate the environment: conda activate verhm
4.	Install all the requirements dependencies mentioned in WorkFlow section

# Datasets

IEMOCAP: In total we are releasing approximately 12 hours of audiovisual data. For each improvised and scripted recording, we provide detailed audiovisual and text information, which consists of the audio and video of both interlocutors, the Motion Capture data of the face, head and hand of one of the interlocutors in each recording, the text trascriptions of the conversation and their word-level, phone-level and syllable-level alignment. Also, for each utterance of the recordings, we provide annotations into categorical and dimensional labels, from multiple annotators. Dataset is avaliable at https://sail.usc.edu/iemocap/iemocap_release.htm

CMU-MOSEI: The Multimodal Corpus of Sentiment Intensity (CMU-MOSI) dataset is a collection of 2199 opinion video clips. Each opinion video is annotated with sentiment in the range [-3,3]. The dataset is rigorously annotated with labels for subjectivity, sentiment intensity, per-frame and per-opinion annotated visual features, and per-milliseconds annotated audio features. Dataset is avaliable at http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/

BIWI: Speech and facial expressions are among the most important channels employed for human communication. During verbal interactions, both the sounds we produce and the deformations which our faces undergo reveal a lot about our emotional states, moods, and intentions. In the future, we forsee computers able to capture those subtle affective signals from the persons they are interacting with, and interfaces able to send such signals back to the human users in the form of believable virtual characters' animations. Dataset is avaliable at https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html

RAVDESS: TThe Ryerson Audio-Visual Database of Emotional Speech and Song Dataset (Ravdees) consists of 7356 files database (total size: 24.8 GB). Two lexically-matched phrases are vocalized in a neutral North American dialect by 24 professional actors (12 female, 12 male). There are calm, happy, sad, angry, terrified, surprised, and disgusted expressions in speech, and there are calm, happy, sad, angry, and frightening expressions in music. Dataset is avaliable at https://datasets.activeloop.ai/docs/ml/datasets/ravdess-dataset/

VOCA: VOCASET is a large collection of audio-4D scan pairs captured from 6 female and 6 male subjects. For each subject, we collect 40 sequences of a sentence spoken in English, each of length three to five seconds. Following, you find the raw scanner data (i.e. raw audio-4D scan pairs), the registered data (i.e. in FLAME topology), and the unposed data (i.e. registered data where effects of global rotation, translation, and head rotation around the neck are removed). See the supplementary video for samples of the dataset. Dataset is avaliable at https://voca.is.tue.mpg.de/download.php

# Step to Run the demo

1.	Download this repository.
2.	Download the emopoi_files folder from the link https://drive.google.com/drive/folders/13aIqYTp4tY5NiusXMcKxyNWiWDp80irP?usp=sharing and put it in the root directory.
3.	Create an 'Images' folder in your project and make subfolders for emotions such as Happy, Worried, Sad, Excited, Exhausted, Frustrated, Bored, and Neutral.
4.	Utilize the 'video_frames.py' file to convert your own or any live video into frames of images, thereby generating a large dataset.
5.	Place the 'image_augmentation.py' and 'emopoi_frontalface_alt.xml' files in each type of image folder. For example, place these files in the "happy" image folder and execute the program. It will detect faces from images, convert them into grayscale, and create new images in the same folder.
6.	Next, create the model. You can copy the code from the 'training_inputs.txt' file, open the terminal in your project folder, paste the code, and hit enter.
7.	Training the model will take approximately 20-25 minutes to complete. Use a large number of datasets for optimal accuracy. 
8.	Upon training completion, two files named 'emopoi_retrained_graph.pb' and 'emopoi_retrained_labels.txt' will be generated.
9.	Finally, run 'emotion_age_gender_recognition.py' (provide the proper path to your video). Based on the recognized emotion, gender, and age group, you will receive personalized POI recommendations suited to the user's preferences.
