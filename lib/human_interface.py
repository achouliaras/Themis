import numpy as np
import os
from pathlib import Path
import imageio
import datetime
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from moviepy.editor import VideoFileClip, clips_array
from gymnasium.spaces import utils as gym_utils
from captum.attr import DeepLift, IntegratedGradients, LRP, Lime, DeepLiftShap, GradientShap,InputXGradient,GuidedBackprop
from captum.attr import GuidedGradCam, Deconvolution, FeatureAblation, Occlusion, FeaturePermutation, ShapleyValueSampling, KernelShap
from captum.attr._utils.lrp_rules import EpsilonRule
from captum.attr._core.lrp import SUPPORTED_LAYERS_WITH_RULES
from captum.attr._utils.visualization import visualize_image_attr
from captum.attr import visualization as viz
from captum.influence import TracInCP, TracInCPFast, TracInCPFastRandProj
import torch
#from torchsummary import summary
import inspect
import matplotlib.pyplot as plt
SUPPORTED_LAYERS_WITH_RULES[nn.Flatten]= EpsilonRule
from collections import Counter
import sys

DEBUG_BATCH_SIZE=2

def topK_indices(seq, k = None, idfun=None): 
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    if k == None:
        return result
    else:
        return result[:k]

class Xplain:
    def __init__(self, agent, action_type, 
                 xplain_action = True, 
                 xplain_state = False, 
                 xplain_reward = False, 
                 xplain_Qvalue= False,
                 checkpoints_dir = None, 
                 replay_buffer = None, 
                 debug = False) -> None:
        self.agent = agent
        self.action_type = action_type

        # Set up which parts will be explained
        self.xplain_action = xplain_action
        self.xplain_state = xplain_state
        self.xplain_reward = xplain_reward
        self.xplain_Qvalue = xplain_Qvalue
        self.checkpoints_dir = checkpoints_dir
        self.replay_buffer = replay_buffer
        self.debug = debug

    def generate_frames(self, sa_t, env, seed, snaps, obs_space):
        # STATE Explanation could be added HERE
        flat_obs_dim = gym_utils.flatdim(obs_space)
        if self.xplain_state:
            replay_dataset = ReplayDataset(self.replay_buffer)
        clips =[]
        clips_raw=[]
        xclips = []
        
        #print(f'SA_T: {sa_t.shape}')
        for i, roll  in enumerate(sa_t):
            if self.debug and i==DEBUG_BATCH_SIZE:
                break

            #print(obs_space.dtype)
            obs = []
            actions = []
            for timestep in roll:
                
                ob = timestep[:flat_obs_dim]
                action = np.uint8(timestep[flat_obs_dim:])
                
                if self.action_type == 'Discrete':
                    # obs space is set as uint8 even if the actual obs is float after normalisation
                    # The unflatten uses the obs space variable to determine the resulting dtype.
                    obs_space.dtype = np.float32
                    ob = gym_utils.unflatten(obs_space,ob)
                
                obs.append(ob)
                actions.append(action)
                
            # Fix dtype to original type to avoid any possible sideffects or incosistencies.
            obs_space.dtype = np.uint8

            env.reset(seed = seed)
            #env.set_state(ob)
            #frames = [env.render()]
            frames=[]
            
            for j in range(roll.shape[0]):
                action = actions[j]
                env.set_state(snaps[i][j])
                next_obs, _, _, _, _ = env.step(action[0])
                frames.append(env.render())

                ### Fix for other envs as well ###

            clips.append(frames)
            clips_raw.append(obs)

            # Create Saliency maps for each trajectory
            if self.xplain_action:
                if self.action_type == 'Discrete':
                    xclip=self.saliency_map(obs, actions, frames)
                    xclips.append(xclip)
                    #print('XAI = ', len(xclip))
            if self.xplain_state:
                xclip=self.influential_states(obs, actions, frames, replay_dataset)
                xclips.append(xclip)
        
        #print(f'Clips {len(clips)}')
        #print(f'Xclips {len(xclips)}')
        
        plot_attrb = False
        if(plot_attrb == True):
            #print(len(xclips[0]))
            print("Xplain= ", xclips[0][0].shape)
            print("Frame= ", frames[0].shape)
            print("State= ", obs[0].shape)

            plt.imshow(xclips[0][0])
            plt.savefig('Clips/exclip.png')
            plt.imshow(frames[0])
            plt.savefig('Clips/frame.png')
            plt.imshow(obs[0])
            plt.savefig('Clips/ob.png')
            plot_attrb = False

        return clips, xclips #clips_raw

    def checkpoints_load_func(self, net, path):
        weights = torch.load(path)
        net.load_state_dict(weights["model_state_dict"])
        return 1.
    
    def influential_states(self, obs, actions, frames, replay_dataset):
        model = self.agent.actor
        model.eval()

        #replay_dataset = DataLoader(dataset=replay_dataset, batch_size=500)
        print(model.trunk)
        tracin_cp_fast = TracInCPFast(
            model=model,
            final_fc_layer=model.trunk[2],
            train_dataset=replay_dataset,  #From Replay buffer
            checkpoints=self.checkpoints_dir,
            checkpoints_load_func=self.checkpoints_load_func,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            batch_size=500,
            vectorize=False,
        )
        
        k = 6
        start_time = datetime.datetime.now()
        
        # Target or NN output dimension mismatch. No idea what is wrong.

        proponents_indices, proponents_influence_scores = tracin_cp_fast.influence(
            (torch.from_numpy(np.array(obs)), torch.tensor(np.array(actions).reshape(-1))), # dim 3-5 cant get it to 4. Maybe change all NN inputs to squeezed forms
            k=k, 
            proponents=True, 
            show_progress= True
        )

        props_idx= proponents_indices.tolist()
        props_idx = [item for sublist in props_idx for item in sublist]
        #print(props_idx,'\n')
        props_idx=sorted(props_idx, key=Counter(props_idx).get, reverse=True)
        #print(props_idx,'\n')
        top_props_idx = topK_indices(props_idx, k)
        #print(top_props_idx,'\n')
        h, w, c = obs[0].shape
        blank = np.zeros(obs[0].shape, dtype=np.uint8)
        window = 8
        mask =[]
        for i in top_props_idx:
            if i-window >= 0 and i+window < len(replay_dataset):
                mask.extend([replay_dataset[i+j][0] for j in range(-window,window+1)])
            elif i-window < 0:
                offset = window-i
                mask.extend([replay_dataset[i+j][0] for j in range(offset-window, window+offset+1)])
            elif i+window >= len(replay_dataset)-1:
                offset = window-len(replay_dataset)+i
                low = -window-offset
                high = window-offset+1
                mask.extend([replay_dataset[i+j][0] for j in range(low, high)])
            mask.extend([blank, blank, blank])
        #print('MASK LENGTH',len(mask))

        return mask

    def saliency_map(self, obs, actions, frames):
        model = self.agent.actor
        action_dim = self.agent.actor_cfg.action_dim
        # Change actor to eval mode to prevent it from learning
        model.eval()
        model.zero_grad()

        #print(model.trunk[4])
        #xai = DeepLift(model) #scale factor zero
        xai = IntegratedGradients(model) #Works
        #xai = LRP(model) #scale factor zero
        #xai = Lime(model) #scale factor zero
        #xai = DeepLiftShap(model) #needs baseline samples
        #xai = GradientShap(model) #needs baseline samples
        #xai = InputXGradient(model) #scale factor zero
        #xai = GuidedBackprop(model) #scale factor zero
        #xai = GuidedGradCam(model,model.cnn[4]) #scale factor zero
        #xai = Deconvolution(model) #scale factor zero
        #xai = FeatureAblation(model) #scale factor zero
        #xai = Occlusion(model) #scale factor zero
        #xai = ShapleyValueSampling(model) #too slow
        #xai = FeaturePermutation(model) #needs multiple samples
        #xai = KernelShap(model) #Works
        
        no_xplain_flag = False
        plot_attrb = True
        mask = []
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        #feature_mask = torch.cat((torch.cat((torch.full((16,16,3), 0), torch.full((16,16,3), 4), torch.full((16,16,3), 8), torch.full((16,16,3), 12)),0),
        #                         torch.cat((torch.full((16,16,3), 1), torch.full((16,16,3), 5), torch.full((16,16,3), 9), torch.full((16,16,3), 13)),0),
        #                         torch.cat((torch.full((16,16,3), 2), torch.full((16,16,3), 6), torch.full((16,16,3), 10), torch.full((16,16,3), 14)),0),
        #                         torch.cat((torch.full((16,16,3), 3), torch.full((16,16,3), 7), torch.full((16,16,3), 11), torch.full((16,16,3), 15)),0)),1)

        for ob, act, frm in zip(obs, actions, frames):
            attribution = xai.attribute(torch.tensor(ob).unsqueeze(0).to(self.agent.device), 
                                      target = int(act[0])# , feature_mask= feature_mask,
                                      ).squeeze(0).cpu().detach().numpy()
            #print(attribution.shape)
            #print(ob.shape)
            #print(frm.shape)
            h, w, _ = attribution.shape
            fig, ax = plt.subplots(figsize=(h*px, w*px))
            fig, ax = visualize_image_attr(attr=attribution, 
                                           original_image=ob, 
                                           method='blended_heat_map', 
                                           plt_fig_axis=(fig, ax), 
                                           use_pyplot=False)
            fig.tight_layout(pad=0)
            canvas = fig.canvas
            canvas.draw()
            plt.close()
            #width, height = canvas.get_width_height()
            s, (width, height) = canvas.print_to_buffer()
            image_array = np.fromstring(s, np.uint8).reshape((height, width, 4))
            
            #mask.append((attribution).astype(np.uint8))
            mask.append((image_array))
            
        
        # print(fig)
        # fig.savefig('Clips/fig')
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
            for i, xclip1 in enumerate(xframes1):
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
            for i, xclip2 in enumerate(xframes2):
                filename = f'TESTxclip2_{i}.mp4'
                with open(p / filename, 'wb') as file2:
                    imageio.mimsave(p / filename, xclip2, fps=20)
                xclips2.append(VideoFileClip(str(p / filename)).margin(10))

        if xflag:
            multiclip = clips_array([[i for i in clips1],[i for i in xclips1],
                                 [i for i in clips2],[i for i in xclips2]])
        else:
            multiclip = clips_array([[i for i in clips1],[i for i in clips2]])

        #multiclip.write_videofile(clipname)
        multiclip.write_videofile(str(p)+'/'+clipname +'.' + format, threads=4, logger = None)
        multiclip.close()

        i=[os.remove(p / f'TESTclip1_{i}.mp4') for i, clip1 in enumerate(frames1)]
        i=[os.remove(p / f'TESTxclip1_{i}.mp4') for i, clip1 in enumerate(xframes1)]
        i=[os.remove(p / f'TESTclip2_{i}.mp4') for i, clip1 in enumerate(frames2)]
        i=[os.remove(p / f'TESTxclip2_{i}.mp4') for i, clip1 in enumerate(xframes2)]

    def generate_paired_clips(self, frames1, xframes1, frames2, xframes2, clipname='TestPairClip', format='mp4'):
        p = Path('Clips')
        p.mkdir(exist_ok=True)
        fps = 6
        xfps = 6
        scale = 256
        xflag = True if len(xframes1) + len(xframes2) > 0 else False
        kargs = { 'macro_block_size': None }

        clips1=[]
        for i, clip1 in enumerate(frames1):
            filename = f'TESTclip1_{i}.mp4'
            with open(p / filename, 'wb') as file1:
                imageio.mimsave(p / filename, clip1, fps=fps, **kargs)
            clips1.append(VideoFileClip(str(p / filename)).margin(10).resize((scale, scale)))
        
        if xflag:
            xclips1=[]
            for i, xclip1 in enumerate(xframes1):
                filename = f'TESTxclip1_{i}.mp4'
                with open(p / filename, 'wb') as file1:
                    imageio.mimsave(p / filename, xclip1, fps=xfps)
                xclips1.append(VideoFileClip(str(p / filename)).margin(10).resize((scale, scale)))

        clips2=[]
        for i, clip2 in enumerate(frames2):
            filename = f'TESTclip2_{i}.mp4'
            with open(p / filename, 'wb') as file2:
                imageio.mimsave(p / filename, clip2, fps=fps, **kargs)
            clips2.append(VideoFileClip(str(p / filename)).margin(10).resize((scale, scale)))
        
        if xflag:
            xclips2=[]
            for i, xclip2 in enumerate(xframes2):
                filename = f'TESTxclip2_{i}.mp4'
                with open(p / filename, 'wb') as file2:
                    imageio.mimsave(p / filename, xclip2, fps=xfps)
                xclips2.append(VideoFileClip(str(p / filename)).margin(10).resize((scale, scale)))

        if xflag:
            clip_combo = zip(clips1,clips2)
            xclip_combo = zip(xclips1,xclips2)
            for i, clips in enumerate(zip(clip_combo, xclip_combo)):
                multiclip = clips_array([clips[0],clips[1]])
                multiclip.write_videofile(str(p)+'/'+clipname + '_' + str(i+1) +'.' + format, threads=4, logger = None)
                multiclip.close()

        else:
            for i, clips in enumerate(zip(clips1,clips2)):
                multiclip = clips_array([clips])
                multiclip.write_videofile(str(p)+'/'+clipname + '_' + str(i+1) +'.' + format, threads=4, logger = None)
                multiclip.close()

        [i.close() for i in clips1]
        [i.close() for i in clips2]
        if xflag:
            [i.close() for i in xclips2]
            [i.close() for i in xclips1]

        i=[os.remove(p / f'TESTclip1_{i}.mp4') for i, clip in enumerate(frames1)]
        i=[os.remove(p / f'TESTclip2_{i}.mp4') for i, clip in enumerate(frames2)]
        if xflag:
            i=[os.remove(p / f'TESTxclip1_{i}.mp4') for i, clip in enumerate(xframes1)]
            i=[os.remove(p / f'TESTxclip2_{i}.mp4') for i, clip in enumerate(xframes2)]

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
    
class ReplayDataset(Dataset):
        def __init__(self, replay_buffer):
            self.idx = replay_buffer.capacity if replay_buffer.full else replay_buffer.idx

            self.states = np.copy(replay_buffer.obses[:self.idx])
            self.actions = np.copy(replay_buffer.actions[:self.idx])
    
        def __len__(self):
            return len(self.states)
        
        def __getitem__(self, idx):
            state, action = self.states[idx], self.actions[idx][0]
            state = torch.from_numpy(state)
            #action = torch.tensor(action)
            #print('Action is: ', action)
            return state, action