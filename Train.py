import os 
import tensorflow as tf

print(tf.__version__)


CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'TEST_IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images', 'test'),
    'TRAIN_IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images', 'train'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

for path in paths.values():
    if not os.path.exists(path):
        os.mkdir(path)

clone_command = "git clone https://github.com/tensorflow/models "+paths['APIMODEL_PATH']


if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    os.system(clone_command)
else:
    print("Tensorflow Object Detection Model found at",paths['APIMODEL_PATH'])
#os.system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . ')
VERIFICATION_SCRIPT = "python3.7 " +os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
#os.system(VERIFICATION_SCRIPT)

import object_detection
url= ('wget '+PRETRAINED_MODEL_URL)
#os.system(url)

move_command = 'mv '+PRETRAINED_MODEL_NAME+'.tar.gz '+paths['PRETRAINED_MODEL_PATH']
#os.system(move_command)

unzip_command = 'cd '+paths['PRETRAINED_MODEL_PATH']+ ' && tar -zxvf ' +PRETRAINED_MODEL_NAME+'.tar.gz'

#os.system(unzip_command)

labels = [{'name':'car', 'id':1}, {'name':'people', 'id':2}, {'name':'rickshaw', 'id':3}, {'name':'animal', 'id':4},{'name':'bike', 'id':5}, {'name':'cycle', 'id':6}, {'name':'red signal', 'id':7}, {'name':'green signal', 'id':8}, {'name':'yellow signal', 'id':9}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
    print('Labelmap Created done')
