import os
import random
import torch
from torch.autograd import grad
from torch import optim
import numpy as np

from model import DQN, DQN_rs, DQN_rs_sig


class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.model_type = args.model_type
    self.game = args.game
    
    if self.model_type == 'DQN':
        self.online_net = DQN(args, self.action_space).to(device=args.device)
        self.target_net = DQN(args, self.action_space).to(device=args.device)
    elif self.model_type == 'DQN_rs': 
        self.online_net = DQN_rs(args, self.action_space).to(device=args.device)
        self.target_net = DQN_rs(args, self.action_space).to(device=args.device)
    elif self.model_type == 'DQN_rs_sig':
        self.online_net = DQN_rs_sig(args, self.action_space).to(device=args.device)
        self.target_net = DQN_rs_sig(args, self.action_space).to(device=args.device)
    
    
    if args.model and os.path.isfile(args.model):
      # Always load tensors onto CPU by default, will shift to GPU if necessary
      self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))
    self.online_net.train()

    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item() 
      # support (1-D of size self.atoms) is broadcasted to returned Q; Returned Q has shape: (batch, self.action_space, self.atoms)
  
  def get_importance(self, state, requires_grad=False):
    if requires_grad:
      state.requires_grad_()
      _, importance = self.online_net.visual_forward(state.unsqueeze(0))
    else:
      with torch.no_grad():
        _, importance = self.online_net.visual_forward(state.unsqueeze(0))
    return importance
  
  def conv_filter_forward(self, state):
    state.requires_grad_()
    return self.online_net.neuron_forward(state.unsqueeze(0))
  
  def v_forward(self, state):
    state.requires_grad_()
    v = self.online_net.vaq_forward(state.unsqueeze(0))[0] #torch.Size([51])
    return v.sum(-1) #(self.atoms) -> scalar
  
  def a_forward(self, state):
    state.requires_grad_()
    return (self.online_net.vaq_forward(state.unsqueeze(0))[1] * self.support).sum(2).squeeze() #(self.action_space)
  
  def q_forward(self, state):
    state.requires_grad_()
    return (self.online_net.vaq_forward(state.unsqueeze(0))[2] * self.support).sum(2).squeeze() #(self.action_space)
  
  def get_saliency_masks(self, state):
    state.requires_grad_()
    importance = self.online_net.forward(state.unsqueeze(0))
    importance = importance.squeeze() # (1,2,7,7) -> (2,7,7)
    importance[0].max().backward(retain_graph=True)
    saliency_0 = torch.empty_like(state.grad.data)
    saliency_0 = state.grad.data.clone()
    state.grad.data.zero_()
    self.online_net.zero_grad() # zeros grads of all module parameters
    
    importance[1].max().backward(retain_graph=False)
    saliency_1 = state.grad.data
    self.online_net.zero_grad()
    return saliency_0, saliency_1
  
  # TODO list:
  # 1. Get intermediate result from model in torch
  # 2. class_split (?) // not do this
  # 3. Using torch.autograd to compute the gradients
  # forward hook
  def get_saliency(self, state, attribution_method, action_idx):
    state.requires_grad_()
    ((self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0]).backward()
    # (self.online_net(state.unsqueeze(0)) * self.support).sum(2) (1,9)

    if attribution_method == 'IG':
      # Implement IG
      class_tensor = self.online_net(state.unsqueeze(0))
      m = 20 # Computing steps
      data = torch.zeros_like(state)
      baseline = torch.zeros_like(state)
      data.requires_grad_()
      IGmap = torch.zeros_like(state)
      for i in range(1, m+1):
            data = (baseline + (i/m) * (state - baseline))
            data.retain_grad()
            ((self.online_net(data.unsqueeze(0)) * self.support).sum(2)[0][action_idx]).backward()
            saliency = data.grad * (1 / m)
            IGmap += saliency
      return IGmap
    elif attribution_method == 'SG':
      # Implement SG
      class_tensor = self.online_net(state.unsqueeze(0))
      m = 20 # Computing steps
      p = 0.2 # percentage of SmoothGrad
      sigma = p * (torch.max(state) - torch.min(state))
      data = torch.zeros_like(state)
      baseline = torch.zeros_like(state)
      data.requires_grad_()
      SGmap = torch.zeros_like(state)
      for i in range(1, m+1):
            data = state + sigma * torch.randn(state.shape)
            data.retain_grad()
            ((self.online_net(data.unsqueeze(0)) * self.support).sum(2)[0][action_idx]).backward(retain_graph=True)
            saliency = data.grad * (1 / m)
            SGmap += saliency
      return SGmap




    self.online_net.zero_grad()
    return state.grad
  
  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, frame=''):
    torch.save(self.online_net.state_dict(), os.path.join(path, self.game+'_'+self.model_type+frame+'_model.pth'))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
