# Voice-Driven 3D Facial Emotion Recognition For Mental Health Monitoring

# About Project

Vocal expressions reflect emotional states, vital for mental health monitoring. Previous methods have difficulty precisely discerning emotions and contents conveyed through speech. Our solution introduces a neural network that disentangles various emotions from voice signals, facilitating the generation of 3D facial expressions.

# Workflow

The first step involves importing essential libraries by installing them on your system. The primary libraries necessary for code execution include:

1. Download the repository
2. Create the conda virtual environment using: `conda create -n verhm python=3.9`
3. Activate the environment: `conda activate verhm`
4. Install all the required dependencies:

- Blender 3.4.1
- ffmpeg 4.4.1
- torch==1.9.0
- torchvision==0.10.0
- torchaudio==0.9.0
- numpy
- scipy==1.7.1
- librosa==0.8.1
- tqdm
- pickle
- transformers==4.6.1
- trimesh==3.9.27
- pyrender==0.1.45
- opencv-python

# Datasets

### IEMOCAP
In total, we are releasing approximately 12 hours of audiovisual data. For each improvised and scripted recording, we provide detailed audiovisual and text information, which consists of the audio and video of both interlocutors, the Motion Capture data of the face, head, and hand of one of the interlocutors in each recording, the text transcriptions of the conversation and their word-level, phone-level, and syllable-level alignment. Also, for each utterance of the recordings, we provide annotations into categorical and dimensional labels, from multiple annotators. The dataset is available at [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm).

### CMU-MOSEI
The Multimodal Corpus of Sentiment Intensity (CMU-MOSI) dataset is a collection of 2199 opinion video clips. Each opinion video is annotated with sentiment in the range [-3,3]. The dataset is rigorously annotated with labels for subjectivity, sentiment intensity, per-frame, per-opinion annotated visual features, and per-millisecond annotated audio features. The dataset is available at [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/).

### BIWI
Speech and facial expressions are among the most important channels employed for human communication. During verbal interactions, both the sounds produced and the deformations that faces undergo reveal much about our emotional states, moods, and intentions. In the future, we foresee computers able to capture those subtle affective signals from the persons they are interacting with and interfaces able to send such signals back to human users in the form of believable virtual characters' animations. The dataset is available at [BIWI](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html).

### AESI
AESI is a new dataset in Greek containing audio recordings of five categorical emotions: anger, fear, joy, sadness, and neutral. The items of the AESI consist of 35 sentences each having content indicative of the corresponding emotion. The resulting data include recordings from 20 participants (12 male, 8 female), which resulted in 696 utterances with a total duration of 27 mins, 51 sec. The dataset is available at [AESI](https://robotics.ntua.gr/aesi-dataset/).

### VOCASET
VOCASET is a large collection of audio-4D scan pairs captured from 6 female and 6 male subjects. For each subject, we collect 40 sequences of a sentence spoken in English, each of length three to five seconds. Following, you find the raw scanner data (i.e. raw audio-4D scan pairs), the registered data (i.e. in FLAME topology), and the unposed data (i.e. registered data where effects of global rotation, translation, and head rotation around the neck are removed). See the supplementary video for samples of the dataset. The dataset is available at [VOCASET](https://voca.is.tue.mpg.de/download.php).

# Training on VOCA Dataset

After downloading the dataset, prepare the data by converting vertices/audio data into .npy/.wav files and storing them in the directories `dataset/vertices_npy` and `dataset/wav`. You can do this by navigating to the dataset directory and executing the script named `process_data.py` using Python.

Then for training, run the `main.py` using Python from the root directory.

# Steps to Run the Demo

1. Download Blender from the [link](https://ftp.nluug.nl/pub/graphics/blender/release/Blender3.4/blender-3.4.1-linux-x64.tar.xz) and put it in the Blender folder in the root directory.
2. Download the pre-trained model from the [link](https://drive.google.com/file/d/1ywEYhMWdxWk9Bqt0UIOdAyYM6v8JUF-K/view?usp=sharing) and put it in the pre-trained folder in the root directory.
3. Run the demo by executing `python run_demo.py --input_voice "input_voice/happy.mp3"`.
