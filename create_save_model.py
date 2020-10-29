import os
import tensorflow as tf

import Mask.model as modellib
from Mask.meta.config.coco_config import CocoConfig

# path of the trained model
dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = dir_path + "/models/"
MODEL_PATH = './models/mask_rcnn_moles_0074.h5'#input("Insert the path of your trained model [ Like models/moles.../mask_rcnn_moles_0030.h5 ]: ")
SAVE_PATH = os.path.normpath(dir_path+'/models/saved_model/1/')
if not os.path.isfile(MODEL_PATH):
    raise Exception(MODEL_PATH + " Does not exists")

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)

# create and instance of config
config = CocoConfig()

# take the trained model
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)

tf.keras.models.save_model(model.keras_model, SAVE_PATH, save_format='tf')

