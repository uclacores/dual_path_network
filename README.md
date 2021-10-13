
# Dual Path Network

## Introduction
The Dual Path Network (DPN) is a neural network architecture for blind symbol decoding and modulation classification. This repository includes the code that accompanies the paper  \[1\]. It includes the code for DPN along with the code for data generation and post processing.

The training data was generated in realtime during training as described in \[1\].  The code for the generator is in this repo. A google drive link to the validation and test sets is available. The weights of the trained network used in the paper are also provided.

There is a known bug in the SNR values and a workaround is provided (See the IMPORTANT WARNING).

\[1\] S. Hanna, C. Dick, and D. Cabric, “Signal Processing Based Deep Learning for Blind Symbol Decoding and Modulation Classification,” arXiv:2106.10543 \[cs, eess\], Jun. 2021, Accessed: Jun. 21, 2021. \[Online\]. Available: http://arxiv.org/abs/2106.10543

## Requirements

* The python packages used with this code are available in requirements.txt (exported from conda). 

* DPN code uses [CuDNNGRU](https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/layers/CuDNNGRU). Hence, an NVidia GPU is required to run the code.

* The validation and test datasets provided in the link are about 10 GB each. A server with  RAM >=32 GB is needed to load them into memory.

Note that these requirements are to replicate the authors' setup.  The code might work with other versions of the packages. 

If no GPU is available the GuDNNGRU layer can be replaced by a regular GRU in frm_nn_zoo_01.py (However, you might need some [workaround](https://stackoverflow.com/a/58810760) to load the weights).

A smaller version of the validation and test dataset can be generated using dataset_creator.ipynb

Also note that data generation is run using multiprocessing with 10 workers. If you have fewer than 10 cores in your setup you might want to reduce this number in  the fit_generator function in 001_d1_train.ipynb

## IMPORTANT WARNING

There is a known bug in the code in the signal generation. A square root is missing in the generation of a noise. As a consequence, the signals generated have twice the required SNR in dB.

For example, when the input value of the SNR in the generator is 10dB, the true SNR of the generated signal is 20dB. 

The workaround is to provide the input SNR value as half of the required value. If you want a signal with 10dB SNR, provide an input value of 5dB.

The bug is in line 30 in frm_dataset_creator.py. However, the datasets and results were generated before the bug was discovered and it was not fixed for backward compatibility.

## Directory Description

### Jupyter Notebooks
**001_d1_train.ipynb**: DPN training

**003_d1_demod_dsp.ipynb**: Decode data using genie algorithm from \[1\]

**004_d1_baseline_nets.ipynb**: Train the SGRU network, which is used as a baseline

**005_d1_demod_dpn.ipynb**: Demodulate the signals using DPN output and store modulation classification (MC) output

**006_d1_compare_demod.ipynb**: Compare demodulation results between Genie and DPN

**008_d1_compare_params.ipynb**: Evaluate frequency and timing offsets

**009_d1_pred_baseline.ipynb**: Generate MC predictions for GRU

**010_d1_mod_class.ipynb**: Compare MC results for DPN and GRU

**013_d1_demod_sample.ipynb**: Plot a signal from the dataset

**020_d1_confusion_matrix.ipynb**: Plot the confusion matrix

**dataset_creator.ipynb**: Code to generate a dataset





### Python Files
**frm_nn_zoo_01.py**: The  code for DPN

**frm_dataset_creator.py**: Code for data generation

**frm_dataset_creator2.py**: Optimized code for data generation

**frm_modulations.py**: Generating signals from different modulations

**frm_modulations_fast.py**: Optimized code for modulations

**frm_train_generator.py**: A keras generator for realtime sample generation

**frm_dataset_loader.py**: Code for reading the dataset from disk

**frm_demod_utils.py**: Functions used for demodulation

**frm_eval_utils.py**: Functions used in the evaulation

**frm_nn_baseline.py**: Neural network code for SGRU

**frm_nn_functions.py**: Keras functions used by DPN

**conf_dataset_1.py**: Configuration file for the datset used in [1]

### Directories
**datasets**: dataset folder (contains google drive link)

**models**: The weights for trained models

**outputs**: Temporary outputs provided

**html**: HTML version of all jupyter notebooks for convenience

**py**: Python version of all jupyter notebooks for convenience

**tmp**: Temporary folder to store the weights

### Other

**requirements.txt**: List of python packages (with version numbers) used with this code. Exported from conda according to  [these instructions](https://stackoverflow.com/a/51293330)

**Readme.md**: This file
