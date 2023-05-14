# Iterative Soft Adversarial Actor-Critic for Safety (ISAACS)

<img src="media/overview.png" alt="drawing" width="88%"/>

[[Webpage]](https://SafeRoboticsLab.github.io/ISAACS) | [[arXiv]](https://arxiv.org/abs/2212.03228)

[Kai-Chieh Hsu](https://kaichiehhsu.github.io/)<sup>1</sup>,
[Duy Phuong Nguyen](https://www.linkedin.com/in/buzinguyen/)<sup>1</sup>,
[Jaime F. Fisac](https://saferobotics.princeton.edu/jaime)

<sup>1</sup>equal contribution in alphabetical order

Princeton University, L4DC'2023

Please raise an issue or reach out at kaichieh or duyn at princenton dot edu if you need help with running the code.


## Installation
1. use either conda or mamda
  + conda
    ```bash
    git submodule update --init --recursive
    conda create -n isaacs python=3.8
    conda activate isaacs
    conda install cuda -c nvidia/label/cuda-11.8.0
    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia/label/cuda-11.8.0
    conda install -c conda-forge suitesparse jupyter notebook omegaconf numpy tqdm jax casadi gym dill plotly shapely wandb matplotlib
    conda install -c cornell-zhang heterocl  
    pip install -e .
    ```
  + mamba
    ```bash
    git submodule update --init --recursive
    mamba create -n isaacs python=3.8
    mamba activate isaacs
    mamba install cuda -c nvidia/label/cuda-11.8.0
    mamba install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia/label/cuda-11.8.0
    mamba install -c conda-forge suitesparse jupyter notebook omegaconf numpy tqdm jax casadi gym dill plotly shapely wandb matplotlib
    conda install -c cornell-zhang heterocl
    pip install -e .
    ```
2. install [pyspline](https://mdolab-pyspline.readthedocs-hosted.com/en/latest/building.html)

## Usage
+ pretrain controller
  ```bash
  python script/pretrain_ctrl.py -cf config/pretrain_ctrl.yaml
  ```
+ pretrain a disturbance policy to be a best response to the controller from the previous stage.
  Note that you need to modify the path in the config. The first version uses the same training script as the one in the third stage. Please see the second version for a cleaner implementation.
  ```bash
  python script/train_isaacs_race_car.py -cf config/isaacs_fix_ctrl.yaml  # ver 1
  python script/pretrain_dstb.py -cf config/pretrain_dstb.yaml  # ver 2
  ```
+ run ISAACS training. Note that you need to modify the path in the config.
  ```bash
  python script/train_isaacs_race_car.py
  ```
+ test safety filter
  1. get [numerical solutions](https://drive.google.com/file/d/1V4bbOa4GYQWypt69N-gT2SjZVaug2Ndn/view?usp=share_link) (treated as oracle in the paper) and put it under `ckpts/odp/results_lw.pkl` (Or, you can modify the path in `config/safety_filter.yaml`)
  2. run below
  ```bash
  python script/test_safety_filter.py -cf config/safety_filter.yaml
  ```

## Citation 

If you find our paper or code useful, please consider citing us with:
```
@inproceedings{hsunguyen2023isaacs,
  title = 	  {ISAACS: Iterative Soft Adversarial Actor-Critic for Safety},
  author =    {Kai-Chieh Hsu and Duy P. Nguyen and Jaime F. Fisac},
  booktitle = {Proceedings of the 5th Conference on Learning for Dynamics and Control},
  year =      {2023},
}
```