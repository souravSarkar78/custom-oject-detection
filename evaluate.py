#This is an incomplete script

import os 
import tensorflow as tf
import time
paths = {}
files = {}
TRAINING_SCRIPT = None

print("evaluating Model...")
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, 
            paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
os.system(command)