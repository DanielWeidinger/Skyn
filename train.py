import numpy as np
import os

import Mask.model as modellib
import Mask.visualize as visualize
from Mask.meta.config.moles_config import MolesConfig
from Mask.meta.serialize_data import serialize_dataset,deserialize_dataset

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIR = dir_path + "/models/"
    COCO_MODEL_PATH = "/Data/mask_rcnn_coco.h5" #os.path.normpath(dir_path + "/Data/mask_rcnn_coco.h5")
    DATA_PATH = "E:\\Data\\" #dir_path + "/Data/data_set.obj"
    DATASET_FILE = "/Data/Data/dataset.obj"
    MODEL_PATH = "./models/mask_rcnn_moles.h5"
    ITERATION = 0
    SHOW_SAMPLES = False

    config = MolesConfig()

    if not os.path.exists(DATASET_FILE):
        print('No preprocessed version found')
        dataset_train, dataset_val = serialize_dataset(DATASET_FILE, DATA_PATH, config)
    else:
        dataset_train, dataset_val = deserialize_dataset(DATASET_FILE)

    # Show some random images to verify that everything is ok
    if SHOW_SAMPLES:
        print(dataset_train.image_ids)
        image_ids = np.random.choice(dataset_train.image_ids, 3)
        for image_id in image_ids:
            image = dataset_train.load_image(image_id)
            mask, class_ids = dataset_train.load_mask(image_id)
            visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Create the MaskRCNN model
    print('creating model...')
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # Use the coco model as start point
    print('loading weights...')
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    # Train the model on the train dataset
    # First only the header layers
    if ITERATION < 30:
        print("training the heads")
        model.train(dataset_train, dataset_val,
                    initial_epoch=ITERATION,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='heads')
    # After all the layers 
    if ITERATION < 90:
        print("training the layers")
        model.train(dataset_train, dataset_val,
                    initial_epoch=ITERATION-30 if ITERATION > 30 else 0,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=90-(30-ITERATION if ITERATION > 30 else 0),
                    layers="all")

    print("Trained finished!")

if __name__ == "__main__":
    main()
