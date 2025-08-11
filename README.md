<div align="center">   

# Common Lane Change Decision Making DL models for Hdd dataset
</div>

Overview
-----

> I am trying to reproduce papers in Lane Change Decision Making which evaluate with highD dataset.

Details
-----
> 

> **A Learning-Based Discretionary Lane-Change Decision-Making Model with Driving Style Awareness**
> - [Paper in arXiv](https://arxiv.org/abs/2010.09533)
> - corresponding to dop(driving operational picture) model in this repo.
> - test result, got overall accuracy of 88.9% in my training and testing process with 2 NVIDIA A40 GPUs. However, there exists a gap between my result and the offical result.

> **Lane-Change-Prediction-LSTM**
> - referenced and modified from [nqyy](https://github.com/nqyy)'s repo.
> - [Github](https://github.com/nqyy/lane-change-prediction-lstm)
> - corresponding to rnn model in this repo. 

Usage
-----
1. Please put the HighD dataset ``*/highd-dataset-v1.0/data/`` in the directory.

2. Run ``python3 ./calculate/get_event_feature.py`` and ``python3 ./calculate/get_time_serues_feature.py `` to process the dataset and the output data for dop model and rnn model will be stored into ``output/`` in pickle format, respectively.

3. Run ``python3 dop_cnn_model/train.py`` and ``python3 rnn_model/rnn_model.py`` to train and test for dop model and rnn model, respectively.

Requirements
------------
Packages installation guide: ``pip3 install -r requirement.txt``
