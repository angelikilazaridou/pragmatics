import os
import json
from subprocess import call

tune_dir = 'grounding/RL/'

for fname in os.listdir(tune_dir):
    if not fname.endswith(".t7"):
        continue
    to_test = tune_dir+fname
    command = "th eval2.lua -model "+to_test+" -batch_size 128 -val_images_use 10000 -noise 0"
    os.system(command)

