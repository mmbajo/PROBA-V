import argparse
import os
import tensorflow as tf
import numpy as np
from skimage import io
from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('__name__')


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--band', type=str)
    opt = parser.parse_args()
    return opt
