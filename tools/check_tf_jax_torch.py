import tensorflow as tf
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print("tf on gpu is available? {}".format(is_cuda_gpu_available))
import jax
def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        raise Exception("Jax does not have GPU support")
print("jax on gpu is available? {}".format(jax_has_gpu()))
import torch
print("torch on gpu is available? {}".format(torch.cuda.is_available()))