import warnings 
warnings.filterwarnings("ignore")

import os, gc
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm, tqdm_notebook
import pandas as pd 
import glob

# print(sys.path)
from mask_functions import rle2mask, mask2rle

DATA_DIR = '/kaggle/input/siim-acr-pneumothorax-segmentation-data/pneumothorax'

# Directory to save logs and trained model
ROOT_DIR = '/kaggle/working'
