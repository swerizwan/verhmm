# Voice-Driven 3D Facial Emotion Recognition For Mental Health Monitoring

# About Project

Vocal expressions reflect emotional states, vital for mental health monitoring. Existing techniques struggle with accuracy. We propose a neural network to untangle emotions in voices, creating 3D facial animations. Our approach surpasses baseline methods, offering improved emotional recognition. This advancement is crucial for enhancing mental health analysis and support.

# Workflow

The first step involves importing essential libraries by installing them on your system. The primary libraries necessary for code execution include:

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

1.	Download the repository
2.	Create the conda virtual environment using: conda create -n verhm python=3.7
3.	Activate the environment: conda activate verhm
4.	Install all the requirements dependencies mentioned in the WorkFlow section

# Datasets

IEMOCAP: In total, we are releasing approximately 12 hours of audiovisual data. For each improvised and scripted recording, we provide detailed audiovisual and text information, which consists of the audio and video of both interlocutors, the Motion Capture data of the face, head, and hand of one of the interlocutors in each recording, the text transcriptions of the conversation and their word-level, phone-level, and syllable-level alignment. Also, for each utterance of the recordings, we provide annotations into categorical and dimensional labels, from multiple annotators. The dataset is avaliable at https://sail.usc.edu/iemocap/iemocap_release.htm

CMU-MOSEI: The Multimodal Corpus of Sentiment Intensity (CMU-MOSI) dataset is a collection of 2199 opinion video clips. Each opinion video is annotated with sentiment in the range [-3,3]. The dataset is rigorously annotated with labels for subjectivity, sentiment intensity, per-frame, per-opinion annotated visual features, and per-milliseconds annotated audio features. The dataset is available at http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/

BIWI: Speech and facial expressions are among the most important channels employed for human communication. During verbal interactions, both the sounds produced and the deformations that faces undergo reveal much about our emotional states, moods, and intentions. In the future, we foresee computers able to capture those subtle affective signals from the persons they are interacting with and interfaces able to send such signals back to human users in the form of believable virtual characters' animations. The dataset is available at https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html

RAVDESS: The Ryerson Audio-Visual Database of Emotional Speech and Song Dataset (Ravdees) consists of 7356 files database (total size: 24.8 GB). Two lexically matched phrases are vocalized in a neutral North American dialect by 24 professional actors (12 female, 12 male). There are calm, happy, sad, angry, terrified, surprised, and disgusted expressions in speech, and there are calm, happy, sad, angry, and frightening expressions in music. The dataset is available at https://datasets.activeloop.ai/docs/ml/datasets/ravdess-dataset/

VOCA: VOCASET is a large collection of audio-4D scan pairs captured from 6 female and 6 male subjects. For each subject, we collect 40 sequences of a sentence spoken in English, each of length three to five seconds. Following, you find the raw scanner data (i.e. raw audio-4D scan pairs), the registered data (i.e. in FLAME topology), and the unposed data (i.e. registered data where effects of global rotation, translation, and head rotation around the neck are removed). See the supplementary video for samples of the dataset. The dataset is available at https://voca.is.tue.mpg.de/download.php

# Training on VOCA Dataset

Prepare the data by converting vertices/audio data into .npy/.wav files and storing them in the directories vocaset/vertices_npy and vocaset/wav. You can do this by navigating to the VOCASET directory and executing the script named process_voca_data.py using Python.

Then for training, execute the main.py using Python from the root directory.

# Step to Run the demo

1.	Download Blender from the link '' and put it in the Blender folder in the root directory.
2.	Download the pre-trained model and put it in the pre-trained folder in the root directory.
3.	Run the demo by python run_demo.py --wav_path "input_voice/happy.mp3"
