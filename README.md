<div align="center">   

# Common Lane Change Decision Making DL/RL models for highD dataset
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

> **Reinforcement Learning methods (DQN / Double DQN / Dueling DQN / PPO)**
> - located in ``rl_models/``.
> - use the same 16-dimensional Feature-A state vector as the RNN model.
> - a data-driven ``LaneChangeEnv`` wraps the pre-processed highD Normal-feature
>   pickles so that any standard RL algorithm can be trained without an external
>   simulator.

Usage
-----
1. Please put the HighD dataset to ``*/highd-dataset-v1.0/data/`` corresponding to the value of ``DATASET_ROOT`` in ``./configs/constant.py``.

2. Modify the ``configs/config.py``, The variable of ``FEATURE_CHOICE`` should be modified to ``CNN_FC``, ``NORMAL`` for dop model and rnn model, respectively. Other Feature construction variables in ``configs/config.py`` also should be modified depends on the user's requirements.

3. Run ``python3 ./calculate/get_event_feature.py`` and ``python3 ./calculate/get_time_series_feature.py`` to process the dataset and the output data for dop model and rnn model will be stored into the folder of ``./output`` in pickle format, respectively.

4. Run ``python3 dop_cnn_model/train.py`` and ``python3 rnn_model/rnn_model.py`` to train and test for dop model and rnn model, respectively.

5. **RL methods** — set ``FEATURE_CHOICE = "Normal"`` in ``configs/config.py`` and run
   ``python3 calculate/get_time_series_feature.py`` first to generate the Normal-feature
   pickles.  Then launch any of the four RL agents:

   ```bash
   python3 rl_models/dqn/train.py          # DQN
   python3 rl_models/double_dqn/train.py   # Double DQN
   python3 rl_models/dueling_dqn/train.py  # Dueling DQN (+ Double DQN update)
   python3 rl_models/ppo/train.py          # PPO with GAE
   ```

   Best-checkpoint weights are saved to ``output/dqn_best.pth``,
   ``output/double_dqn_best.pth``, ``output/dueling_dqn_best.pth``, and
   ``output/ppo_best.pth`` respectively.

   **RL models overview:**

   | Model | Key idea |
   |---|---|
   | DQN | Q-learning with experience replay and a target network |
   | Double DQN | Decouples action selection (online net) from action evaluation (target net) to reduce overestimation bias |
   | Dueling DQN | Splits Q into value stream V(s) and advantage stream A(s,a) for better generalisation across actions |
   | PPO | On-policy policy-gradient with clipped surrogate objective and GAE advantage estimation |

6. Run ``python3 extract_conscend.py`` to generate a ConScenD-style scenario dataset from highD. The script exports:
   - scenario metadata JSON files in ``./output/conscend/metadata/``
   - OpenSCENARIO ``.xosc`` files in ``./output/conscend/scenarios/``
   - straight-road OpenDRIVE ``.xodr`` files in ``./output/conscend/roads/``

7. Optional arguments for the ConScenD workflow:
   - ``python3 extract_conscend.py --recordings 1 2 3`` to process selected recordings only
   - ``python3 extract_conscend.py --dataset-root /path/to/highd-dataset-v1.0``
   - ``python3 extract_conscend.py --output-root /path/to/output/conscend``

ConScenD reproduction notes
-----

The new extraction pipeline follows the paper title and available methodological details for
**The ConScenD Dataset: Concrete Scenarios from the highD Dataset According to ALKS Regulation UNECE R157 in OpenX**.
It adds a ConScenD-style scenario export workflow on top of the existing highD readers in this repository.

Implemented scenario logic:
- ALKS speed-domain filtering at ``60 km/h`` equivalent
- lane-change detection from ``laneId`` transitions
- cut-in classification using adjacent following / alongside vehicles before the maneuver
- non-lane-change classification for free-driving and car-following scenarios
- OpenSCENARIO and OpenDRIVE export for each extracted scenario

This implementation is intended to be reproducible with the public highD dataset and repository code,
while remaining explicit that some paper details were inferred from the regulation and related OpenX references.

Python version
--------------
python_version == 3.12.3

Requirements
------------
Packages installation guide: ``pip3 install -r requirement.txt``
Anaconda was recommended here.

