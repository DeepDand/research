#!/bin/sh

sudo apt-get update

sudo -H apt-get -y install vim

sudo -H apt-get -y install libatlas-dev

sudo -H apt-get -y install gcc
sudo -H apt-get -y install gfortran

sudo -H apt-get -y install python-pip
sudo -H apt-get -y install python-dev
sudo -H apt-get -y install python-numpy
sudo -H apt-get -y install python-scipy
sudo -H apt-get -y install python-matplotlib

sudo -H apt-get -y install openssl
sudo -H apt-get -y install libffi-dev
sudo -H apt-get -y install build-essential

sudo -H pip install --upgrade pip
sudo -H pip install -U setuptools
sudo -H pip install --upgrade numpy scipy wheel cryptography

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl
#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
sudo -H pip install --upgrade $TF_BINARY_URL

sudo -H pip install  nvidia-ml-py

wget -O- http://bit.ly/glances | /bin/bash

sudo -H apt-get install tree
sudo -H apt-get install screen
sudo -H pip install image


git config --global user.name "Pablo Rivas"
git config --global user.email pablorp80@gmail.com


