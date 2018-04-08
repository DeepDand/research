#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u autoencoder.py 2>&1 | tee result_8192_4096_2.txt

