![alt text](https://github.com/achouliaras/Themis/blob/main/logo.png)
## Install the following modules to use Themis

```
Pytorch
Captum
Gymnassium
Minigrid
Hydra
termcolor
moviepy
Matplotlib
Pandas
```

## Supported Gym environments

- MuJoCo (eg. domain=Control, env=Humanoid-v4)
- Atari (eg. domain=ALE, env=Breakout-v5)
- Box2d (eg. domain=Box2d, env=Humanoid-v4)
- Minigrid (eg. domain=Minigrid, env=DistShift1-v0)
- BabyAI (eg. domain=BabyAI, env=GoToRedBallGrey-v0)

You can manually add more environments as long as they follow Gym format.

## Clip Sampling Options

- Uniform Sampling        (feed_type=0)
- Disagreement Sampling   (feed_type=1)
- Entropy Sampling        (feed_type=2)
- K Center                (feed_type=3)
- K Center + Disagreement (feed_type=4)
- K Center + Entropy      (feed_type=5)

### SAC and unsupervised pre-training

Experiments can be executed with the following scripts:

```
./themis_pretrain.sh 
./themis_train.sh 
```

Edit the files accordigly to specify changes in the experiment configuration.

## Learn reward
To run experiment using a learned reward model set the flag '''learn_reward''' to True. Otherwise the environment reward will be used.

## Run experiments on human teachers
Be sure to change the flag '''human_teacher''' to True.
The method '''get_labels''' in the file '''reward_model.py''' contains the logic to generate clips ang receive input from the user. Explore the available tools from the '''lib/human_interface.py'''.

## Use explainable techniques
To use the explainable techniques currently supported set either the '''xplain_action''' or '''xplain_state''' flag to True. Refer to '''lib/human_interface.py''' if you want to add more.

## Run experiments on synthetic teachers

Themis is based on BPref, so it incorporates the same logic toward the synthetic teachers. To tweak the synthetic teacher tweak the relevant parameters in '''config/train_themis.py''':

```
teacher_beta: rationality constant of stochastic preference model (default: -1 for perfectly rational model)
teacher_gamma: discount factor to model myopic behavior (default: 1)
teacher_eps_mistake: probability of making a mistake (default: 0)
teacher_eps_skip: hyperparameters to control skip threshold (\in [0,1])
teacher_eps_equal: hyperparameters to control equal threshold (\in [0,1])
```

Synthetic teacher examples:

`Oracle teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Mistake teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0.1, teacher_eps_skip=0, teacher_eps_equal=0)

`Noisy teacher`: (teacher_beta=1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Skip teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0.1, teacher_eps_equal=0)

`Myopic teacher`: (teacher_beta=-1, teacher_gamma=0.9, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Equal teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0.1)

