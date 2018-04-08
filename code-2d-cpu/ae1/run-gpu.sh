#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python -u autoencoder.py 2>&1 | tee result_2.txt

