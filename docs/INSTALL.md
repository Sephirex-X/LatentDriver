## Installation
**Step 1:** Create an python environment for LatentDriver:
```shell
conda create --name ltdriver python=3.10
conda activate ltdriver 
```
**Step 2:** Install compatable tensorflow+jax+torch according to your cuda \& cudnn (*VERY IMPORTANT*, please follow [JAX](https://jax.readthedocs.io/en/latest/installation.html)). \
My environment is cuda=12.1+cudnn=8.8 and CUDA driver version is 12.0. If your environment is simular, you can use the following command:
```shell
# install tf
pip install tensorflow==2.15.0 
# install jax
pip install jax==0.4.10 jaxlib==0.4.10+cuda12.cudnn88 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# install torch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
**Step 3:** Clone the codebase, and compile this codebase: 
```shell
# Clone LatentDriver codebase
git clone https://github.com/Sephirex-X/LatentDriver

# install the dependencies
pip install -r requirements.txt

# Compile the customized CUDA/C++ codes in LatentDriver codebase
python setup.py install
```
You can check if your tensorflow+jax+torch environment is correctly set up by running:
```shell
python tools/check_tf_jax_torch.py
```
Then you are all set.

**Note:** We have modified the original waymax codebase for metric calcaution, so please use the code in this repo. \
 If you have installed waymax following [this](https://github.com/waymo-research/waymax) on your device, you can skip the JAX related installation but have to uninstall waymax by `pip uninstall waymax`