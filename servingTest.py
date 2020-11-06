import json
import os
import numpy as np
import cv2
import requests

import Mask.model as modellib
import Mask.visualize as visualize
from Mask.meta.config.coco_config import CocoConfig

np.set_printoptions(threshold=np.inf)

# path of the trained model
dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = dir_path + "/models/"
MODEL_PATH = './models/mask_rcnn_moles_0074.h5'#input("Insert the path of your trained model [ Like models/moles.../mask_rcnn_moles_0030.h5 ]: ")
if not os.path.isfile(MODEL_PATH):
    raise Exception(MODEL_PATH + " Does not exists")

# create and instance of config
config = CocoConfig()

# take the trained model
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

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

images = [img]


# Mold inputs to format expected by the neural network
molded_images, image_metas, windows = model.mold_inputs(images)

# Validate image sizes
# All images in a batch MUST be of the same size
image_shape = molded_images[0].shape
for g in molded_images[1:]:
    assert g.shape == image_shape,\
        "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

# Anchors
anchors = model.get_anchors(image_shape)
anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)

# Run object detection
headers = {"content-type": "application/json"}
data = json.dumps({
    'signature_name': "predict",
    'instances': [
            {
                'input_image:0': molded_images.tolist()[0],
                'input_image_meta:0': image_metas.tolist()[0],
                'input_anchors:0': anchors.tolist()[0]
            }
        ]
    })
res = requests.post('http://86.56.135.139:31967/v1/models/skyn:predict', data=data, headers=headers)
decoded = json.loads(res.content.decode("utf-8"))['predictions'][0]

detections = np.array([decoded['mrcnn_detection/Reshape_1:0']])
mrcnn_mask = np.array([decoded['mrcnn_mask/Reshape_1:0']])

# Process detections
results = []
for i, image in enumerate(images):
    final_rois, final_class_ids, final_scores, final_masks =\
        model.unmold_detections(detections[i], mrcnn_mask[i],
                               image.shape, molded_images[i].shape,
                               windows[i])
    results.append({
        "rois": final_rois,
        "class_ids": final_class_ids,
        "scores": final_scores,
        "masks": final_masks,
    })

r = results[0]
print(r['scores'])
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

