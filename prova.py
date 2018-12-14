import tensorflow as tf
import numpy as np
from pyglet.gl import *

print(np.atleast_2d(np.array([1,2,3])))
print(tf.concat([np.atleast_2d(np.array([1,2,3])), np.atleast_2d(np.array([4]))],1))