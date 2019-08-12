"""
Mask R-CNN
Train on the toy Mitochondria dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 Mitochondria.py train --dataset=/path/to/Mitochondria/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 Mitochondria.py train --dataset=/path/to/Mitochondria/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 Mitochondria.py train --dataset=/path/to/Mitochondria/dataset --weights=imagenet

    # Apply color splash to an image
    python3 Mitochondria.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 Mitochondria.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import matplotlib.pyplot as plt

import scipy.ndimage as ndimage 
from skimage import measure   

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from imgaug import augmenters as iaa
import imgaug as ia

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class MitochondriaConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Mitochondria"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2


    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Mitochondria

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 148//2 #4826//6 1776 1776//10  7104 39072//20 148//8  25900//5  24130//6

    # Skip detections with < 90% confidence

    VALIDATION_STEPS = 17//2 #204 4488//20 17//8 2975//28  2975//8     2975//5 2750//6

    BACKBONE = "resnet50"

    
    IMAGE_CHANNEL_COUNT = 3

    #MEAN_PIXEL = np.array([140.7])

    #IMAGE_RESIZE_MODE = "crop"
    #IMAGE_MIN_DIM = 768
    #IMAGE_MAX_DIM = 768



    MEAN_PIXEL = np.array([140.7,140.7,140.7])

    LEARNING_RATE = 0.0001

    POST_NMS_ROIS_TRAINING = 1000//1
    POST_NMS_ROIS_INFERENCE = 1000//1
    TRAIN_ROIS_PER_IMAGE =  200//1
    RPN_TRAIN_ANCHORS_PER_IMAGE =  256//1

    #IMAGE_MIN_DIM = 256
    #IMAGE_MAX_DIM = 256

    RPN_ANCHOR_SCALES = (32, 64, 128)

    RPN_NMS_THRESHOLD = 0.5

    PRE_NMS_LIMIT = 3000


    DETECTION_MAX_INSTANCES = 25

    MAX_GT_INSTANCES = 22
    # Max number of final detections

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.99

    # Non-maximum suppression threshold for detection


    ROI_POSITIVE_RATIO = 0.2

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    USE_MINI_MASK = True  #True
    MINI_MASK_SHAPE = (56, 56) 

    TOP_DOWN_PYRAMID_SIZE = 256

    WEIGHT_DECAY = 0.0001

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1,
        "rpn_bbox_loss": 1,
        "mrcnn_class_loss": 1,
        "mrcnn_bbox_loss": 1,
        "mrcnn_mask_loss": 1
    }




############################################################
#  Dataset
############################################################

class MitochondriaDataset(utils.Dataset):

    def load_Mitochondria(self, dataset_dir, subset):
        """Load a subset of the Mitochondria dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Mitochondria", 1, "Mitochondria")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)


        for filename in os.listdir(dataset_dir+'/orig'):
            self.add_image(
                "Mitochondria",
                image_id=filename,
                path=os.path.join(dataset_dir+'/orig', filename))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Mitochondria dataset image, delegate to parent class.
        image_info = self.image_info[image_id]

        nuevo=os.path.dirname(os.path.dirname(image_info['path']))
        nuevo=os.path.join(nuevo, 'mask')
        path, filename = os.path.split(image_info['path'])
        final=os.path.join(nuevo, filename)
        
        mask = skimage.io.imread(final)
        mask[mask<150]=0
        mask[mask>150]=255
        contours = measure.find_contours(mask, 0.6)
        lista=[]
        #TOTALMENTE PRESCINDBLE
        for n, contour in enumerate(contours):
            if(contour.shape[0]>0):
                centro=np.mean(contour,axis=0)
                if(mask[int(centro[0]),int(centro[1])]!=0):
                    if(False==np.array_equal(contour[0,:],contour[-1,:])):
                        w=((contour[0,:]+contour[-1,:])/2).astype(int)
                        if(mask[w[0],w[1]]!=0):
                            if((contour[0,0]==contour[-1,0] or contour[0,1]==contour[-1,1])):
                                lista.append(contour)
                            else:
                                if(len(contour)>30):
                                    if(mask[w[0],w[1]-1]==255 and mask[w[0]+1,w[1]]==255 and mask[w[0]-1,w[1]]==255 and 255==mask[w[0],w[1]+1]):
                                        lista.append(contour)
                    else:
                        lista.append(contour)

        if(len(lista)==0):
            return np.empty([mask.shape[0], mask.shape[1],1]), np.zeros([1], dtype=np.int32)

        lista3=np.empty([mask.shape[0], mask.shape[1],len(lista)])

        for i in range(len(lista)):
            if(False==np.array_equal(lista[i][0,:],lista[i][-1,:])):
                #centro=lista[i].sum(axis=0)/lista[i].shape[0]
                centro=(lista[i][0,:]+lista[i][-1,:])/2
                #rad=(min(np.sqrt(np.square(lista[i]-centro).sum(axis=1)))+max(np.sqrt(np.square(lista[i]-centro).sum(axis=1))))/2
                rad=max(np.sqrt(np.square(lista[i]-centro).sum(axis=1)))
                lista3[:,:,i]=create_circular_mask(mask.shape[0],mask.shape[1],center=[centro[1],centro[0]],radius=rad)*mask

            else:
        
                r_mask = np.zeros_like(mask, dtype='bool')

        
                r_mask[np.round(lista[i][:, 0]).astype('int'), np.round(lista[i][:, 1]).astype('int')] = 1

                r_mask = ndimage.binary_fill_holes(r_mask)
                lista3[:,:,i]=r_mask*mask

        lista3[np.isnan(lista3)] = 0

        lista3[lista3<150]=0
        lista3[lista3>150]=1

        lista4=np.sum(lista3,axis=2)


        for i in range(0,lista3.shape[-1]):
            c = measure.find_contours(lista3[:,:,i], 0.5)
            if(len(c)>1):
                lista3[:,:i]=lista3[:,:,i][lista4==2]=0

        return lista3, np.ones([lista3.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Mitochondria":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    

def create_circular_mask(h, w, center=None, radius=None):

        if center is None: # use the middle of the image
            center = [int(w/2), int(h/2)]
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

def train(model):
    """Train the model."""
    # Training dataset.
    
    
    dataset_train = MitochondriaDataset()
    dataset_train.load_Mitochondria(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MitochondriaDataset()
    dataset_val.load_Mitochondria(args.dataset, "val")
    dataset_val.prepare()


    augmentation=iaa.Sequential([
        iaa.OneOf([iaa.Affine(rotate=0),
                   iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Flipud(0.5),
        iaa.Fliplr(0.5),
        iaa.SomeOf((0, 2), [
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},mode='reflect',order=0),
        iaa.Affine(rotate=(-30,30),mode='reflect',order=0),
        iaa.CropAndPad(percent=(-0.10, 0.10))])])
    
    model.train(dataset_train, dataset_val,
                learning_rate=0.0001,
                epochs=50,
                layers='all',
                augmentation=augmentation)

    model.train(dataset_train, dataset_val,
                learning_rate=0.0001/10,
                epochs=100,
                layers='all',
                augmentation=augmentation)

    model.train(dataset_train, dataset_val,
                learning_rate=0.0001/100,
                epochs=150,
                layers='all',
                augmentation=augmentation)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Mitochondrias.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Mitochondria/dataset/",
                        help='Directory of the Mitochondria dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    #import time
    #seconds = time.time()
    #print("Seconds since epoch =", seconds) 

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MitochondriaConfig()
    else:
        class InferenceConfig(MitochondriaConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            #IMAGE_RESIZE_MODE = "pad64"
            #RPN_NMS_THRESHOLD = 0.6
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "bueno":
        weights_path = '/home/iecheverria/Desktop/PRACTICAS/Mask_RCNN-master/logs/mitochondria20190809T1111/mask_rcnn_mitochondria_0075.h5'
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes  , "conv1"
        
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

    
    #sseconds = time.time()
    #print('Tiempoooooooo')
    #print((sseconds-seconds)/3600)
