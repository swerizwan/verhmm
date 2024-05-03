# Voice-Driven 3D Facial Emotion Recognition For Mental Health Monitoring

## About the Project

Vocal expressions are significant indicators of emotional states, crucial for mental health monitoring. However, existing methods often struggle to accurately separate the emotions and contents conveyed through speech alone. Our solution introduces a neural network that effectively disentangles various emotions from voice signals, enabling the generation of precise 3D facial expressions.

## Workflow

To begin, ensure you have the following essential libraries installed on your system:

- Pytorch 1.9.0
- CUDA 11.3
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

## Installation

Follow these steps to set up the project:

1. Download the repository.
2. Create a conda virtual environment using: `conda create -n verhm python=3.7`.
3. Activate the environment: `conda activate verhm`.
4. Install all required dependencies mentioned in the Workflow section.

## Datasets

Our project utilizes several datasets for training and testing purposes:

- **IEMOCAP**: Provides approximately 12 hours of audiovisual data with detailed annotations and alignments. [Dataset Link](https://sail.usc.edu/iemocap/iemocap_release.htm)
- **CMU-MOSEI**: A collection of opinion video clips with sentiment annotations. [Dataset Link](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
- **BIWI**: Contains speech and facial expression data crucial for understanding emotional states. [Dataset Link](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html)
- **AESI**: A Greek dataset featuring recordings of five categorical emotions. [Dataset Link](https://robotics.ntua.gr/aesi-dataset/)
- **VOCASET**: Offers audio-4D scan pairs for emotion analysis. [Dataset Link](https://voca.is.tue.mpg.de/download.php)

## Training Steps

After downloading the dataset, prepare the data by converting vertices/audio data into .npy/.wav files and store them in the directories `dataset/vertices_npy` and `dataset/wav`. Then, for training, run `main.py` from the root directory using Python.

## Running the Demo

To run the demo, follow these steps:

1. Download Blender from [Blender Official Website](https://www.blender.org/download/), and place it in the Blender folder within the root directory.
2. Download the pre-trained model from [Pre-Trained Model Link](https://drive.google.com/file/d/1ywEYhMWdxWk9Bqt0UIOdAyYM6v8JUF-K/view?usp=sharing) and put it in the `pre-trained` folder in the root directory.
3. Run the demo by executing `run_demo.py` with the desired input voice using Python: 
