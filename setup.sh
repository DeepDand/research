#!/bin/sh
wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz
tar xvzf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc

sudo apt-get update
sudo apt-get upgrade

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

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
sudo -H pip install --upgrade $TF_BINARY_URL

sudo -H pip install  nvidia-ml-py

wget -O- http://bit.ly/glances | /bin/bash

sudo -H apt-get install tree
sudo -H apt-get install screen
sudo -H pip install image


git config --global user.name "Pablo Rivas"
git config --global user.email pablorp80@gmail.com


