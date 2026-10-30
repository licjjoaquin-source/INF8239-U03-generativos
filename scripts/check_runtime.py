import platform

import tensorflow as tf

print("Python/OS:", platform.python_version(), platform.platform())
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
