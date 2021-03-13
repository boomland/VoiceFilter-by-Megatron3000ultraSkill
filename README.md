# :mega: VoiceFilter by Megatron3000ultraSkill 
This is Skoltech ML Final Project, devoted to 
+ Separation of one speaker's voice from noise and other voices 
+ Removing speaker's voice from the background noise

### Tasks
+ Replicate the resutls of [VoiceFilter paper](https://arxiv.org/pdf/1810.04826.pdf)
+ Analyze if filtered audio samples improve quality for speech recognition
+ Reverse the model so it keeps all but one

### :rocket: Launch the model 
We used **p2s.16xlarge.8** for the following scenario.
#### Requirements
We tested the code on Python 3.6 with PyTorch 1.0.1. Other packages can be installed by:
  <pre> pip install -r requirements.txt</pre>
#### Datasets
For training and testing phases we used [VoxCeleb1 datasets](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html), which consist of celebrities utterances, extracted from videos uploaded to YouTube. <br> We selected for training all the utterances from 30 speakers for testing all the utterances from 10 speakers.

#### Data Preparement
Perform STFT for train and test files before training by:
  <pre>python generator.py -c [config.yaml file] -d [VoxCeleb1 directory (should ends with <i>aac</i>)] -o [output directory]</pre>
We got 100,000 train and 1000 test samples. 
#### The Model 

![GitHub Logo](/model.png)

The model consists of 
+ Speaker Encoder (3-layer LSTM), which produces a speaker embedding from audio samples of the target speaker 
+ VoiceFilter (we used the variant with 8 convolutional layers, 1 LSTM layer, and 2 fully connected layers, each with ReLU activations except for the last layer, which has a sigmoid activation). 

To **reimplement** the model run:
  <pre>python trainer.py -c [config.yaml file] -e [path of embedder pt file] -m [create a name for the model]</pre>

To get **the results**  run:
<pre> python res.py -c [config.yaml file] -e [path of embedder pt file] --checkpoint_path [chkpt/name/chkpt_{step}.pt] </pre>
#### Results
After 1.2k steps we got the following results:
![GitHub Logo](/res.png)

Metrics comparing to **paper results**, where [LibriSpeech dataset](https://www.openslr.org/12) was used:

| Median SDR (Source to Distortion Ratio)  | Paper | Ours |
| ------------- | ------------- |------------- |
| Before VoiceFilter | 2.5  | 1.6 |
| After VoiceFilter  | 12.6 | 4.8 |

[![Watch the video](/mixed.mp4)

### Acknowledgements  
Based on https://github.com/mindslab-ai/voicefilter.
### Team Members 
+ [Mikhail Filitov](https://github.com/lll-phill-lll)
+ [Yaroslav Pudyakov](https://github.com/boomland)
+ [Ildar Gabdrakhmanov](https://github.com/KotShredinger)
+ [Anita Soloveva](https://github.com/aniton)
