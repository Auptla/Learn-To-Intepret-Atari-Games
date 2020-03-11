import argparse
from datetime import datetime
import random
import torch

from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test
import seaborn; seaborn.set()

import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--model-type', type=str, default='DQN', help='DQN or DQN with attention')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(80e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between logging status')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--saliency-thresh', type=float, default=0., metavar='S', help='Saliency zero-off threshold')
parser.add_argument('--folder', type=str, default='', metavar='FOLDER', help='Folder that contains trained models')
parser.add_argument('--multiply', action='store_true', help='Multiply attention with screen')
parser.add_argument('--suffix', type=str, default='', help='video name suffix to differentiate versions')
parser.add_argument('--max', action='store_true', help='Max saliency over 4 channels')
parser.add_argument('--mask', action='store_true', help=' If yes, use saliency to produce mask; if no, use raw saliency for multiplication')
parser.add_argument('--heatmap', action='store_true', help=' If yes, visualize saliency as heatmap')
parser.add_argument('--channel', type=str, default='', help='0 or 1')
parser.add_argument('--original', action='store_true', help='')
parser.add_argument('--attribution-method', type=str, default='', help='IG, SG or other attribute method')
parser.add_argument('--gaussian-noise', action='store_true', help='add gaussian noise to the saliency map')
parser.add_argument('--action', type=int, default=0, help='0-4')

# Setup
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(random.randint(1, 10000))
  torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
  print('using GPU')
else:
  print('using CPU')
  args.device = torch.device('cpu')


# Environment
env = Env(args)
env.eval()
img_h, img_w, _ = env.ale.getScreenRGB()[:, :, ::-1].shape


# Agent
PRETRAINED_MODEL = './results/'+args.game+'_'+args.model_type+'_model.pth'
print(PRETRAINED_MODEL)
dqn = Agent(args, env)
dqn.online_net.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=args.device))
dqn.target_net.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=args.device))

dqn.eval()  # Set DQN (online network) to evaluation mode
mem = ReplayMemory(args, args.memory_capacity)
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# movie recorder
prefix = './visualize/'+args.game+'/'
fourcc = cv2.VideoWriter_fourcc(*'XVID')

if args.multiply and args.mask:
    out = cv2.VideoWriter(prefix+args.game+'_'+args.model_type+'_'+args.folder+'_seed'+str(args.seed)+'_thresh'+str(args.saliency_thresh)+'_maskMultiply_'+ args.attribution_method + '_action' + str(args.action) + args.suffix+'.avi', fourcc, 10.,
        (img_w*2+1*1, img_h), isColor=True)
        #(img_w*3+2*1, img_h), isColor=True)
    if args.channel == '0':
        out = cv2.VideoWriter(prefix+args.game+'_'+args.model_type+'_'+args.folder+'_seed'+str(args.seed)+'_thresh'+str(args.saliency_thresh)+'_maskMultiply_0'+args.suffix+'.avi', fourcc, 8., (img_w, img_h), isColor=True)
    elif args.channel == '1':
        out = cv2.VideoWriter(prefix+args.game+'_'+args.model_type+'_'+args.folder+'_seed'+str(args.seed)+'_thresh'+str(args.saliency_thresh)+'_maskMultiply_1'+args.suffix+'.avi', fourcc, 8., (img_w, img_h), isColor=True)
elif args.multiply and not args.mask:
    out = cv2.VideoWriter(prefix+args.game+'_'+args.model_type+'_'+args.folder+'_seed'+str(args.seed)+'_thresh'+str(args.saliency_thresh)+'_rawMultiply'+args.suffix+'.avi', fourcc, 8., (img_w*3+2*1, img_h), isColor=True)
    if args.channel == '0':
        out = cv2.VideoWriter(prefix+args.game+'_'+args.model_type+'_'+args.folder+'_seed'+str(args.seed)+'_thresh'+str(args.saliency_thresh)+'_rawMultiply_0'+args.suffix+'.avi', fourcc, 8., (img_w, img_h), isColor=True)
    elif args.channel == '1':
        out = cv2.VideoWriter(prefix+args.game+'_'+args.model_type+'_'+args.folder+'_seed'+str(args.seed)+'_thresh'+str(args.saliency_thresh)+'_rawMultiply_1'+args.suffix+'.avi', fourcc, 8., (img_w, img_h), isColor=True)
elif args.heatmap:
    out = cv2.VideoWriter(prefix+args.game+'_'+args.model_type+'_'+args.folder+'_seed'+str(args.seed)+'_thresh'+str(args.saliency_thresh)+'_heatmap'+args.suffix+'.avi', fourcc, 8., (img_w*3+2*1, img_h), isColor=True)
else:
    out = cv2.VideoWriter(prefix+args.game+'_'+args.model_type+'_'+args.folder+'_seed'+str(args.seed)+'_thresh'+str(args.saliency_thresh)+'_'+args.suffix+'.avi', fourcc, 8., (img_w*3+2*1, img_h), isColor=True)
#(img_w*3+20+5, img_h)

if args.original:
    out = cv2.VideoWriter(prefix+args.game+'_'+args.model_type+'_'+args.folder+'_seed'+str(args.seed)+'_thresh'+str(args.saliency_thresh)+'_original'+args.suffix+'.avi', fourcc, 8., (img_w, img_h), isColor=True)

# Test performance
done = True

if(args.attribution_method):
    print("The attribute method is:", args.attribution_method)
else:   
    print("No attribution specified!")

while True:
  if done:
    state, reward_sum, done = env.reset(), 0, False

  action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
  saliency = dqn.get_saliency(state, args.attribution_method, args.action)

  #print('raw state shape is {}'.format(state.shape)) # (4,84,84)
  #print('saliency shape is {}'.format(saliency.shape)) # (4,84,84)
  if args.max:
    saliency = torch.max(saliency, dim=0)[0]
    #print(saliency_0.max()) # 7.7252e-08
    #print('saliency shape is {}'.format(saliency.shape)) # (84, 84)
  else:
    saliency = saliency[-1] 
  
  state = env.ale.getScreenRGB()[:, :, ::-1].astype(np.uint8)
  # ===========normalize saliency========
  saliency = torch.abs(saliency)
  saliency -= saliency.min()
  
  #print(saliency_0.max()) # 1.5421e-07, 0.
  saliency /= torch.clamp(torch.max(saliency), min=1e-8)
  saliency[saliency<args.saliency_thresh] = 0
  saliency = saliency.data.cpu().numpy().astype(np.float32)
  
  saliency = cv2.resize(saliency, (img_w,img_h), cv2.INTER_CUBIC) #INTER_LINEAR, LANCZOS4: 8X8


  if(args.gaussian_noise):
      noise = 0.05 * (np.max(saliency) - np.min(saliency)) * np.random.standard_normal(size=saliency.shape) 
      saliency += noise
  
  if args.heatmap:
    #(210,160,3)
    temp = np.zeros((img_h, img_w, 3),'float32')
    temp[:,:, 2] = saliency
    saliency = (temp*255).astype(np.uint8)
    saliency += (saliency[:,:,2]==0)[:,:,np.newaxis] * state
    del temp
  
  elif not args.multiply:
    saliency = (cv2.cvtColor(saliency,cv2.COLOR_GRAY2RGB)*255).astype(np.uint8)
    alpha = 0.7
    saliency = cv2.addWeighted(saliency, alpha, state, 1 - alpha, 0)    # src1, src2, dst

  else:
    saliency = np.expand_dims(saliency, axis=2)    #(210, 160, 1)
    if args.mask:
        saliency = ((saliency>0.1) * state).astype(np.uint8)
    else:
        saliency = (saliency * 1.5 * state).astype(np.uint8)
  
  if args.channel == '':
      gap = (np.ones((img_h, 1, 3), 'float32')*255).astype(np.uint8)
      #saliency = np.concatenate((saliency_0, gap, saliency_1), 1)
      output = np.concatenate((state, gap, saliency), 1)
      
  elif args.channel == '0':
      output = saliency
  elif args.channel == '1':
      output = saliency
  
  if args.original:
    output = state
  
  out.write(output) # Write out frame to video np.uint8(output)
  
  state, reward, done = env.step(action)  # Step
  reward_sum += reward
  #if args.render:
  #  env.render()
  
  if done:
    break

env.close()
out.release()


print('Reward: ' + str(reward_sum) )


