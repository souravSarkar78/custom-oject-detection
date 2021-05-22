import os 
import tensorflow as tf
import time

epoch = 4000
batch_size = 6
print(tf.__version__)

def give_space(n=2):
    for i in range(n):
        print(" ")

if not os.path.exists("Tensorflow"):
    os.mkdir("Tensorflow")

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
    'IMAGE_PATH': os.path.join('images'),
    'TEST_IMAGE_PATH': os.path.join('images', 'test'),
    'TRAIN_IMAGE_PATH': os.path.join('images', 'train'),
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


#Downloading Tensorflow Object-detection API

clone_command = "git clone https://github.com/tensorflow/models "+paths['APIMODEL_PATH']
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    print("Downloading Tensorfloe Object-detection-API")
    os.system(clone_command)
else:
    print("Tensorflow Object Detection Model found at",paths['APIMODEL_PATH'])
give_space()

# Installing TF Object detection API
print("Installing TF Object detection API ..... ") 
protobuff_command = 'cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .'
os.system(protobuff_command)
give_space()


print("Verifing System.....")
give_space(1)
#############os.system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . ')
VERIFICATION_SCRIPT = "python3.7 " +os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
os.system(VERIFICATION_SCRIPT)
print("System Verified Succesfully....")
give_space()

import object_detection

print("Downloading Pretrained Model....")
url= ('wget '+PRETRAINED_MODEL_URL)
os.system(url)

print("Succesfully Downloaded Pretrained Model..")

move_command = 'mv '+PRETRAINED_MODEL_NAME+'.tar.gz '+paths['PRETRAINED_MODEL_PATH']
os.system(move_command)

unzip_command = 'cd '+paths['PRETRAINED_MODEL_PATH']+ ' && tar -zxvf ' +PRETRAINED_MODEL_NAME+'.tar.gz'

os.system(unzip_command)

labels = [{'name':'car', 'id':1}, {'name':'people', 'id':2}, {'name':'rickshaw', 'id':3}, {'name':'animal', 'id':4},{'name':'bike', 'id':5}, {'name':'cycle', 'id':6}, {'name':'red signal', 'id':7}, {'name':'green signal', 'id':8}, {'name':'yellow signal', 'id':9}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
    print('Labelmap Created.....')

give_space()

if not os.path.exists(files['TF_RECORD_SCRIPT']):
    print("Downloading TF_Record Script.....")
    git_link_command = 'git clone https://github.com/nicknochnack/GenerateTFRecord '+paths['SCRIPTS_PATH']
    os.system(git_link_command)

    print("TF_Record Script Downloaded..!!")

give_space(1)
print("Generating TF_Record...........")
tf_record_train = 'python '+files['TF_RECORD_SCRIPT']+' -x '+paths['TRAIN_IMAGE_PATH']+' -l '+files['LABELMAP']+' -o '+os.path.join(paths['ANNOTATION_PATH'], 'train.record') 
tf_record_test = 'python '+files['TF_RECORD_SCRIPT']+' -x '+paths['TEST_IMAGE_PATH']+' -l '+files['LABELMAP']+' -o '+os.path.join(paths['ANNOTATION_PATH'], 'test.record') 
os.system(tf_record_train)
os.system(tf_record_test)
print("TF_Record Generated succsfully ...........")

give_space()


if os.name =='posix':
    cp_command = 'cp '+os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')+' '+os.path.join(paths['CHECKPOINT_PATH'])
    os.system(cp_command)

print("Copied Pipeline config.")
time.sleep(2)
give_space()

from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

print("Modifing config file....")

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = batch_size
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]


config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)  

print("Config modified..")
give_space(1)

print("Hold on ... Training is Starting....")
give_space(1)

TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], epoch)
give_space()
st = time.time()
os.system(command)
give_space(1)
print("Training took: ", (time.time()-st)/60, " minutes")

give_space(5)

print("evaluating Model...")
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
give_space(1)
os.system(command)

