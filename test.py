import os
import numpy as np
import cv2

import Mask.model as modellib
import Mask.visualize as visualize
from Mask.meta.config.coco_config import CocoConfig

np.set_printoptions(threshold=np.inf)

# path of the trained model
dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = dir_path + "/models/"
MODEL_PATH = './models/mask_rcnn_moles.h5'#input("Insert the path of your trained model [ Like models/moles.../mask_rcnn_moles_0030.h5 ]: ")
if not os.path.isfile(MODEL_PATH):
    raise Exception(MODEL_PATH + " Does not exists")

# create and instance of config
config = CocoConfig()

# take the trained model
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)



# path of Data that contain Descriptions and Images
path_data = './test_data/3.jpg'#input("Insert the path of Data [ Link /home/../ISIC-Archive-Downloader/Data/ ] : ")
if not os.path.exists(path_data):
    raise Exception(path_data + " does not exists")

# background + (malignant , benign)
class_names = ["BG", "malignant", "benign"]

#meta = json.load(open(path_data+"Descriptions/"+filename))
img = cv2.imread(path_data)
img = cv2.resize(img, (128, 128))
if img is None:
    exit(1)

# predict the mask, bounding box and class of the image
r = model.detect([img])[0]
print(r['scores'])
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
