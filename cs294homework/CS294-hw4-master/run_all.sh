#!/bin/bash

rm -rf ./log/*

CUDA_VISIBLE_DEVICES='' python3 main.py

# cd trpo
# CUDA_VISIBLE_DEVICES='' python3 trpo_main.py 
