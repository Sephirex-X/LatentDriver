
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# dis able preallocate GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
# enable 64 bit computation for python indice
# jax.config.update("jax_enable_x64", True)
# set the default device to GPU if available for jax
# jax.config.update("jax_default_device", jax.devices()[-1])
# set GPU memory growth for tensorflow
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')