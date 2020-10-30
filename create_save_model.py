import os
import tensorflow as tf
import cv2

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

#model.keras_model.save(SAVE_PATH)
builder = tf.python.saved_model.builder.SavedModelBuilder(SAVE_PATH)

#get signature from compiled model
signature = tf.python.saved_model.signature_def_utils.predict_signature_def(inputs={x.name: x for x in model.keras_model.input}, outputs={x.name: x for x in model.keras_model.output})

import keras
builder.add_meta_graph_and_variables(keras.backend.get_session(), tags=[tf.python.saved_model.tag_constants.SERVING], signature_def_map={"predict": signature})
builder.save()
