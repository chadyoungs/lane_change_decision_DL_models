<div align="center">   

# Common Lane Change Decision Making DL models for highD dataset
</div>

Overview
-----

> I am trying to reproduce papers in Lane Change Decision Making which evaluate with highD dataset.
> - [Paper in arXiv for highD Dataset](https://arxiv.org/abs/1810.05642)

Details
-----
> 

> **A Learning-Based Discretionary Lane-Change Decision-Making Model with Driving Style Awareness**
> - [Paper in arXiv](https://arxiv.org/abs/2010.09533)
> - corresponding to dop(driving operational picture) model in this repo.
> - test result, got overall accuracy of 88.9% in my training and testing process with 2 NVIDIA A40 GPUs. However, there exists a gap between my result and the offical result.

> **Lane-Change-Prediction-LSTM**
> - referenced and modified from [nqyy](https://github.com/nqyy)'s repo.
> - [Github repo](https://github.com/nqyy/lane-change-prediction-lstm)
> - corresponding to rnn model in this repo. 

Usage
-----
1. Please put the HighD dataset to ``*/highd-dataset-v1.0/data/`` corresponding to the value of ``DATASET_ROOT`` in ``./configs/constant.py``.

2. Modify the ``configs/config.py``, The variable of ``FEATURE_CHOICE`` should be modified to ``CNN_FC``, ``NORMAL`` for dop model and rnn model, respectively. Other variables should be modified depends on the user's requirements.

3. Run ``python3 ./calculate/get_event_feature.py`` and ``python3 ./calculate/get_time_series_feature.py`` to process the dataset and the output data for dop model and rnn model will be stored into the folder of ``./output`` in pickle format, respectively.

4. Run ``python3 dop_cnn_model/train.py`` and ``python3 rnn_model/rnn_model.py`` to train and test for dop model and rnn model, respectively.

Python version
--------------
python_version == 3.12.3

Requirements
------------
Packages installation guide: ``pip3 install -r requirement.txt``
Anaconda was recommended here.


