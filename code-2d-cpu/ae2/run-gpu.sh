#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python -u autoencoder.py 2>&1 | tee result_8192_2.txt

