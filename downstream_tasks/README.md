## To run Gender Recognition task evaluate:

Install requirements.

`python3 gender_recognition.py`

After finish of script there will be 3 files with plots: male, female and general recognition.
To get not smooth version of plots comment `x{}, y{} = get_new(x{}, y{})` in code (follow the instructions in script)

## To run Speech-to-text  task evaluate:
`virtualenv -p python3 $HOME/tmp/deepspeech-venv/`
`source $HOME/tmp/deepspeech-venv/bin/activate`

`pip3 install deepspeech`

`curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm`
`curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer`

run `python3 speech_to_text.py`

After finish of script there will be plot with 3 lines target, noisy and filtered.
To get not smooth version of plots comment `x{}, y{} = get_new(x{}, y{})` in code (follow the instructions in script)
