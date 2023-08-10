import numpy as np
import os
from pathlib import Path
import imageio
from moviepy.editor import VideoFileClip, clips_array
from gymnasium.spaces import utils as gym_utils
from captum.attr import DeepLift
#from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule
import torch
from torchsummary import summary


class Xplain:
    def __init__(self, agent, action_type, 
                 xplain_action = True, 
                 sample_states = False, 
                 xplain_reward = False, 
                 xplain_Qvalue= False) -> None:
        self.agent = agent
        self.action_type = action_type

        # Set up which parts will be explained
        self.xplain_action = xplain_action
        self.sample_states = sample_states
        self.xplain_reward = xplain_reward
        self.xplain_Qvalue = xplain_Qvalue

    def generate_frames(self, sa_t, env, seed, obs_space):
        # STATE Explanation could be added HERE
        flat_obs_dim = gym_utils.flatdim(obs_space)

        clips =[]
        xclips = []

        for reps in sa_t:
            env.reset(seed = seed)
            env.render()
            #print(reps[0].shape)
            #print(gym_utils.flatdim(obs_space))
            obs = reps[0][:flat_obs_dim]
            if self.action_type == 'Discrete':
                obs = gym_utils.unflatten(obs_space,obs)
            env.set_state(obs)
            frames = [env.render()]

            if self.xplain_action:
                x_obs = []
                x_actions = []

            for roll in reps:
                action = np.uint8(roll[flat_obs_dim:])

                #print(obs_space.shape)
                # Obs + action for Saliency map
                if self.xplain_action:
                    x_obs.append(obs)
                    x_actions.append(action)

                next_obs, _, _, _, _ = env.step(action)
                frames.append(env.render())
                obs = next_obs

            clips.append(frames)

            # Use create Saliency maps for each trajectory
            if self.xplain_action:
                if self.action_type == 'Discrete':
                    xclips.append(self.saliency_map(x_obs, x_actions))
                    print('XAI = ',xclips)

        return clips, xclips

    def saliency_map(self, obs, action):
        model = self.agent.actor
        # Change actor to eval mode to prevent it from learning
        model.eval()

        print(torch.tensor(obs[0]).unsqueeze(0).shape)
        summary(model, obs[0].shape)

        xai = DeepLift(model)
        mask = []
        for ob, act in zip(obs,action):
            mask.append(xai.attribute(torch.tensor(ob).unsqueeze(0), target = act))
        
        # Change actor back to train mode to continue training
        model.train()

        return mask

    def generate_merged_clip(self, frames1, xframes1, frames2, xframes2, clipname='TestMergedClips', format='mp4'):
        p = Path('Clips')
        p.mkdir(exist_ok=True)
        xflag = True if len(xframes1) + len(xframes2) > 0 else False

        clips1=[]
        for i, clip1 in enumerate(frames1):
            filename = f'TESTclip1_{i}.mp4'
            with open(p / filename, 'wb') as file1:
                imageio.mimsave(p / filename, clip1, fps=20)
            clips1.append(VideoFileClip(str(p / filename)).margin(10))
        
        if xflag:
            xclips1=[]
            for i, xclip1 in enumerate(frames1):
                filename = f'TESTxclip1_{i}.mp4'
                with open(p / filename, 'wb') as file1:
                    imageio.mimsave(p / filename, xclip1, fps=20)
                xclips1.append(VideoFileClip(str(p / filename)).margin(10))

        clips2=[]
        for i, clip2 in enumerate(frames2):
            filename = f'TESTclip2_{i}.mp4'
            with open(p / filename, 'wb') as file2:
                imageio.mimsave(p / filename, clip2, fps=20)
            clips2.append(VideoFileClip(str(p / filename)).margin(10))
        
        if xflag:
            xclips2=[]
            for i, xclip2 in enumerate(frames2):
                filename = f'TESTxclip2_{i}.mp4'
                with open(p / filename, 'wb') as file2:
                    imageio.mimsave(p / filename, xclip2, fps=20)
                xclips2.append(VideoFileClip(str(p / filename)).margin(10))

        if xflag:
            multiclip = clips_array([[[i for i in clips1],[i for i in xclips1]],
                                 [[i for i in clips2],[i for i in xclips2]]])
        else:
            multiclip = clips_array([[i for i in clips1],[i for i in clips2]])

        #multiclip.write_videofile(clipname)
        multiclip.write_videofile(str(p)+'/'+clipname +'.' + format, threads=4, logger = None)

        i=[os.remove(p / f'TESTclip1_{i}.mp4') for i, clip1 in enumerate(frames1)]
        i=[os.remove(p / f'TESTclip2_{i}.mp4') for i, clip1 in enumerate(frames2)]

    def generate_paired_clips(self, frames1, frames2, clipname='TestPairClip', format='mp4'):
        p = Path('Clips')
        p.mkdir(exist_ok=True)

        clips1=[]
        for i, clip1 in enumerate(frames1):
            filename = f'TESTclip1_{i}.mp4'
            with open(p / filename, 'wb') as file1:
                imageio.mimsave(p / filename, clip1, fps=20)
            clips1.append(VideoFileClip(str(p / filename)).margin(10))
        
        clips2=[]
        for i, clip2 in enumerate(frames2):
            filename = f'TESTclip2_{i}.mp4'
            with open(p / filename, 'wb') as file2:
                imageio.mimsave(p / filename, clip2, fps=20)
            clips2.append(VideoFileClip(str(p / filename)).margin(10))

        for i, clips in enumerate(zip(clips1,clips2)):
            multiclip = clips_array([clips])
            multiclip.write_videofile(str(p)+'/'+clipname + '_' + str(i+1) +'.' + format, threads=4, logger = None)

        i=[os.remove(p / f'TESTclip1_{i}.mp4') for i, clip1 in enumerate(frames1)]
        i=[os.remove(p / f'TESTclip2_{i}.mp4') for i, clip1 in enumerate(frames2)]

    def get_batch_input_keyboad(self, input_size):
        # Get human input from keyboard as a string of the individual choices
        print("\nType 1 for the top video, 2 for the bottom and SPACE if equal")
        choice_list = list(input('Type anything else to discard all preferences\n'))
        while len(choice_list)!=input_size:
            print(f'Wrong Input size. Provide {input_size} preferences\n')
            print("Type 1 for the top video, 2 for the bottom and SPACE if equal\n")
            choice_list = list(input('Type anything else to discard all preferences'))

        labels=[]
        for choice in choice_list:
            if choice == '1':
                labels.append([1])
            elif choice == '2':
                labels.append([0])
            elif choice == ' ':
                labels.append([-1])
            else:
                return []
        return labels

    def get_input_keyboad(self, input_size):
        # Get human input from keyboard for each individual choice
        print("\nType 1 for the left video, 2 for the right and SPACE if equal")
        print('If you wish to discard all preferences type "skip"\n')
        if input() == 'skip':
            return []
        labels=[]

        while len(labels) < input_size:
            choice = input(f'Type Choice for clip No: {len(labels)+1}\n')
            if choice == '1':
                labels.append(1)
            elif choice == '2':
                labels.append(0)
            elif choice == ' ':
                labels.append(-1)
            else:
                print('Wrong input. Try again') 
        return labels