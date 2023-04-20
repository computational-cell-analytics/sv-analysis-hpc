# Vesicle Segmentation and Parameter Extraction: HPC Prototype

There are two scripts: 
- segment_vesicles.py: Run the vesicle segmentation. Expects the path to the tomogram and the model weights as input (both are available in the ownCloud folder: `sample_data/57K_TEM_tomogram.mrc`, `sample_data/3D_UNet_for_Vesicle_Segmentation.zip`. The corresponding filepaths need to be changed in the script). Writes the segmentation as a hdf5 file.
- extract_parameters.py: Extract parameters from the vesicle segmentation (here: coordinates and radii). Expects the path to the segmentation as input and writes out the parameters in a csv file.

The dependencies needed for running the scripts can be installed either via conda or pip and are listed in
- environment.yaml (for conda, create a new environment with these environments via `conda env create -f environment.yaml`)
- requirements.txt (for pip)

Note: in the case of conda this will create an environment with a cpu version of PyTorch.


## Training

The script `train_supervised.py` is for training a 3D U-Net on tomograms (stored as mrc files) and vesicle labels (stored as h5 files).
The file `environment_training.yaml` can be used to set up a conda environment with all dependencies needed for running this script.
This environment will install a version of PyTorch compiled with CUDA, so that training can be run on the GPU.
Note that you may need to change the CUDA version to match the nvidia driver version installed on your system (line 12 in the environment file).
