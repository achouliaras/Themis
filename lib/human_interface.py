import os
from pathlib import Path
import imageio
from moviepy.editor import VideoFileClip, clips_array

def generate_frames(sa_t, env, seed):
    # STATE Explanation could be added HERE
    clips =[]
    for reps in sa_t:
        env.reset(seed = seed)
        env.render()
        obs = reps[0][:env.observation_space.shape[0]]
        env.set_state(obs)
        frames = []
        for roll in reps:
            action = roll[env.observation_space.shape[0]:]
            _, _, _, _, _ = env.step(action)
            frames.append(env.render())
        clips.append(frames)

    return clips

def generate_merged_clip(frames1, frames2, clipname='TestMergedClips', format='mp4'):
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

    multiclip = clips_array([[i for i in clips1],[i for i in clips2]])
    multiclip.write_videofile(clipname)
    multiclip.write_videofile(str(p)+'/'+clipname +'.' + format, threads=4, logger = None)

    i=[os.remove(p / f'TESTclip1_{i}.mp4') for i, clip1 in enumerate(frames1)]
    i=[os.remove(p / f'TESTclip2_{i}.mp4') for i, clip1 in enumerate(frames2)]

def generate_paired_clips(frames1, frames2, clipname='TestPairClip', format='mp4'):
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

def get_batch_input_keyboad(input_size):
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

def get_input_keyboad(input_size):
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