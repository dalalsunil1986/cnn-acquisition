# convolutional neural networks


## Download the Mario Kart gameplay videos
2 videos are in this folder: 
* fast (Mario trying to go as fast as possible)
* slow (Mario trying to be a slow poke)
* put the videos in a folder (mk_videos) inside this repo folder.
* videos available here: https://www.dropbox.com/sh/7k3m5s3ql4hfx7u/AACMQAr6fRbiexLWQDGSmV0Za?dl=0


## Setup the conda environment
* in Terminal run `conda env create -f mario-kart.yml`
* `source activate mario-kart`

## Acquire training data and test data from videos and train the model
* `python model-new.py`
The last command starts the data acquisition and pipes the iterator through our cnn

### Receiving updates from upstream
Just fetch the changes and merge them into your project with git.

## Contact
Hit me up here on github

## License
MIT
