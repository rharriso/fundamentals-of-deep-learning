from datatools import input_data
mnist = input_data.read_data_sets("./data-sets/mnist", one_hot=True)

import tensorflow as tf
import time, shutil, os
