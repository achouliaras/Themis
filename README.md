## Install

```
conda env create -f conda_env.yml
pip install -e .[docs,tests,extra]
cd custom_dmcontrol
pip install -e .
cd custom_dmc2gym
pip install -e .
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
pip install pybullet
```

## Supported Gym environments

- MuJoCo (eg. domain=Control env=Humanoid-v4)
- Atari (eg. domain=ALE env=Breakout-v5)
- Box2d (eg. domain=Box2d env=Humanoid-v4)
- Minigrid (eg. domain=Minigrid env=DistShift1-v0)
- BabyAI (eg. domain=BabyAI env=GoToRedBallGrey-v0)

You can manually add more environments as long as they follow the Gym format.

## Clip Sampling Options

- Uniform Sampling        (feed_type=0)
- Disagreement Sampling   (feed_type=1)
- Entropy Sampling        (feed_type=2)
- K Center                (feed_type=3)
- K Center + Disagreement (feed_type=4)
- K Center + Entropy      (feed_type=5)

### SAC + unsupervised pre-training

Experiments can be reproduced with the following:

```
./run_pretrain.sh 
./run_train.sh 
```

## Run experiments on human teachers
Be sure to change the flag '''human_teacher''' to True.
The method '''get_labels''' in the file '''reward_model.py''' contains the logic to generate clips ang receive input from the user. Explore the available tools from the '''human_interface.py'''

## Run experiments on synthetic teachers

The tools from BPref to tweak the synthetic teacher are supported and work in the same way:

```
teacher_beta: rationality constant of stochastic preference model (default: -1 for perfectly rational model)
teacher_gamma: discount factor to model myopic behavior (default: 1)
teacher_eps_mistake: probability of making a mistake (default: 0)
teacher_eps_skip: hyperparameters to control skip threshold (\in [0,1])
teacher_eps_equal: hyperparameters to control equal threshold (\in [0,1])
```

In B-Pref, there are the following teachers:

`Oracle teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Mistake teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0.1, teacher_eps_skip=0, teacher_eps_equal=0)

`Noisy teacher`: (teacher_beta=1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Skip teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0.1, teacher_eps_equal=0)

`Myopic teacher`: (teacher_beta=-1, teacher_gamma=0.9, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Equal teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0.1)

