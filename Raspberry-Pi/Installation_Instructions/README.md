# Detailed installation instructions from scratch:

***

## I) Install vanilla/recommended Raspbian OS on Raspberry 4 through official installer.
***
## II) Install all requirements below:
***
### 1) Pytorch (use PyTorch4Raspbery.sh script)

### 2) Tensorflow 2 (use Tensorflow4Raspberry script)

### 3) Keras 2 (use Keras4Raspberry.sh script)

### 4) Numpy (use Numpy4Raspberry.sh script)

## III) Install the rest of the required packages by using the commands below:
***
### pip3 install dill
### pip3 install --user secrets
### pip3 install tqdm

***

### pip3 install pandas
### pip3 install matplotlib
### pip3 install ipython

***

### pip3 install music21
### pip3 install mido
### pip3 install pretty_midi
### pip3 install fluidsynth
### pip3 install pyfluidsynth
### pip3 install midi2audio

***

### sudo apt autoremove
### sudo shutdown -r 0

***

## IV) Install the drivers for RaspiAudio MIC+ board:
***
### sudo wget -O - mic.raspiaudio.com | sudo bash
### sudo shutdown -r 0
### sudo wget -O - test.raspiaudio.com | sudo bash

***
## V) Install startup script (rc.local) from etc directory.
***

### sudo cp rc.local /etc/rc.local
### sudo shutdown -r 0

***
## If you have done everything correctly, you should be good to go and you should hear the welcome voice prompts after the last reboot :)

## Enjoy! :)

***
## Useful Links:

### RaspberryPi OS download and install via NOOBS: https://www.raspberrypi.org/downloads/
### PyTorch: https://www.spinellis.gr/blog/20200317/index.html
### Numpy: https://python-forum.io/Thread-Installing-Numpy-on-Raspberry-Pi
### Keras: https://medium.com/@abhizc/installing-latest-tensor-flow-and-keras-on-raspberry-pi-aac7dbf95f2
### IPython: https://blog.domski.pl/ipython-notebook-server-on-raspberry-pi/

### https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
### https://stackoverflow.com/questions/62433716/how-to-solve-importerror-libhdf5-serial-so-103-cannot-open-shared-object-file
### https://docs.h5py.org/en/stable/build.html#source-installation
### https://github.com/lhelontra/tensorflow-on-arm
### https://github.com/lhelontra/tensorflow-on-arm/releases
### https://www.tensorflow.org/install/install_raspbian
***

#### Project Los Angeles
#### Tegridy Code 2020
