# :mega: VoiceFilter by Megatron3000ultraSkill 
This is Skoltech ML Final Project, devoted to 
+ Separation of one speaker's voice from noise and other voices 
+ Removing speaker's voice from the background noise

### Tasks
+ Replicate the resutls of [VoiceFilter paper](https://arxiv.org/pdf/1810.04826.pdf)
+ Analyze if filtered audio samples improve quality for speech recognition
+ Reverse the model so it keeps all but one

### Launch the model 
#### Requirements
We tested the code on Python 3.6 with PyTorch 1.0.1. Other packages can be installed by:
  <pre> pip install -r requirements.txt</pre>
#### Datasets
We used [LibriSpeech datasets](http://www.openslr.org/12/) for training: <p>train-clean-100.tar.gz</p>, for testing: <p>dev-clean.tar.gz</p>.

### Team Members 
+ Mikhail Filitov 
+ Yaroslav Pudyakov
+ [Ildar Gabdrakhmanov](https://github.com/KotShredinger)
+ [Anita Soloveva](https://github.com/aniton)
