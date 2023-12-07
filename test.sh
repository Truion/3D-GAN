#!/usr/bin/env bash

# download models
gdown "https://drive.google.com/uc?export=download&id=1taDMcaFS2MyteGhMLrtjlp64dMOB3cu3" -O Models_GAN/generator.pth

# run test
python test.py --generator_path Models_GAN/generator.pth
