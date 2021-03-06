import os
import random
import ast
import gym
import numpy as np
from collections import namedtuple, deque
from numpy.random import default_rng
#from IPython.display import clear_output
from itertools import count
from pympler import asizeof

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import time

from memory import ReplayMemory, Transition

SEED = 42


class Agent:
    def __init__(self, env, config, model_class, device):
        self.env = env
        self.seed=SEED
        self.rng = default_rng(SEED)
        #self.number_states = env.observation_space.n
        self.number_actions = env.action_space.n
        self.config = config
        self.step_count = 0
        self.device = device
        self.memory = None
        self.loss = None
        self.model_class = model_class

    def compile(self):
        """ Initialize the model """
        n_actions = self.number_actions

        
        #self.model = self.model_class(n_actions).to(self.device)
        self.model = self.model_class(
                n_states=len(self.config['simulation']['states']),
                outputs=n_actions,
                hid_dim=self.config['model']['hid_dim']
                ).to(self.device)
        self.model.to(self.device)
        #self.target_model = self.model_class(n_actions).to(self.device)
        self.target_model = self.model_class(
                n_states=len(self.config['simulation']['states']),
                outputs=n_actions,
                hid_dim=self.config['model']['hid_dim']
                ).to(self.device)
        self.target_model.eval() # Evaluation mode. train = false
        self.optimizer = optim.Adam(self.model.parameters(), 
                lr=self.config["training"]["learning_rate"])
        #self.optimizer = optim.Adam(self.model.parameters(), 
                #lr=self.config["training"]["learning_rate"])

    def _get_epsilon(self,episode):
        eps_min = self.config["epsilon"]["min_epsilon"]
        eps_max = self.config["epsilon"]["max_epsilon"]
        eps_decay = self.config["epsilon"]["decay_epsilon"]
        epsilon = eps_min + (eps_max-eps_min)*np.exp(-episode/eps_decay)
        return epsilon
                
    def _get_action(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #predicted = self.model(torch.tensor([state], device=self.device))
            predicted = self.model(torch.tensor(state, device=self.device))
            action = predicted.max(1)[1]
        return action.item()

    def _choose_action(self, state, epsilon):
        if self.rng.uniform() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self._get_action(state)
        return action

    def _adjust_learning_rate(self, episode):
        # TODO
        if True: 
            a = None

    def _train_model(self):
        """ Calculate loss and update weights """
        if len(self.memory) < self.config["training"]["batch_size"]:
            return
        transitions = self.memory.sample(self.config["training"]["batch_size"])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # >>> zip(*[('a', 1), ('b', 2), ('c', 3)]) === zip(('a', 1), ('b', 2), ('c', 3))
        # [('a', 'b', 'c'), (1, 2, 3)]
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Compute prediced Q values - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        predicted_q_value = self.model(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute the expected Q values
        # t.max(1) will return largest column value of each row.
        # First column on max result is max element for each row.
        # Target networks perceived Q values for next states following training
        # networks "choice" of action.
        next_state_values = self.target_model(next_state_batch).max(1)[0]
        expected_q_values = (~done_batch * next_state_values * self.config["rl"]["gamma"]) + reward_batch
        #expected_q_values = (~done_batch * next_state_values * self.config["rl"]["gamma"]) + reward_batch

        # Compute Huber loss.
        self.loss_function = nn.SmoothL1Loss()
        loss = self.loss_function(predicted_q_value, expected_q_values.unsqueeze(1))

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp(-1,1)
        self.optimizer.step()


    def _update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _remember(self, state, action, next_state, reward, done):
        #self.memory.push(torch.tensor([state], device=self.device),
        self.memory.push(
            torch.tensor(state, device=self.device),
            torch.tensor(action, device=self.device, dtype=torch.long),
            torch.tensor(next_state, device=self.device),
            torch.tensor(reward, device=self.device),
            torch.tensor(done, device=self.device, dtype=torch.bool))

    def fit(self, path, verbose:bool=False):
        # Stateless counterpart for nn.SmoothL1Loss()
        self.loss = F.smooth_l1_loss
        self.memory= ReplayMemory(self.config["rl"]["memory_capacity"])
        epsilon = 1
        vid_h = self.config['simulation']['video']['height']
        vid_w = self.config['simulation']['video']['width']
        vid_size = vid_h*vid_w*3
        if verbose: print(self.config['simulation']['states'])
        for i_episode in range(self.config["rl"]["num_episodes"]):

            if verbose: print("Episode: ", i_episode)

            # TODO: Is this reasonable for initial state. i.e. first memory?
            state = self.env.reset()[vid_size:]
            if state.size == 0:
                state = [0 for s in self.config['simulation']['states']]
            if i_episode >= self.config["training"]["warmup_episode"]:
                epsilon = self._get_epsilon(
                        i_episode - self.config["training"]["warmup_episode"])

            avg_r = 0
            for step in count():
                action = self._choose_action(state, epsilon)
                import pdb; pdb.set_trace()
                #next_state, reward, done, _ = self.env.step(action)
                _, reward, done, info = self.env.step(action)

                if done and reward is None: reward = 0
                avg_r += reward

                if info != '' and info is not None: 
                    info = info.replace("true", "True")
                    info = info.replace("false", "False")
                    info = ast.literal_eval(info)
                    next_state = [info[s] for s in info if s in self.config['simulation']['states']]
                else: 
                    # TODO: Is this reasonable for initial next_state. i.e. first memory?
                    next_state = [0 for s in self.config['simulation']['states']]

                self._remember(state, action, next_state, reward, done)
                state = next_state
                if i_episode >= self.config["training"]["warmup_episode"]:
                    self._train_model()
                    #self._adjust_learning_rate(i_episode - self.config["training"]["warmup_episode"] + 1) # TODO
                    done = (step == self.config["rl"]["max_steps_per_episode"] - 1) or done
                else:
                    # Justify choice of magic number!
                    done = (step == 5 * self.config["rl"]["max_steps_per_episode"] -1) or done 
                if done: 
                    if verbose:
                        #import pdb; pdb.set_trace()
                        print(f"Average reward: {avg_r/(step+1):.2f}")
                    break

    
            # Update target network every C episode
            if i_episode % self.config["rl"]["target_model_update_freq"] == 0:
                self._update_target()

            # Save weights, Necessary?
            if i_episode % self.config["training"]["save_freq"] == 0:
                #self.save()
                torch.save(self.model.state_dict(), path)
                print("Saved")

            print("Mem:", asizeof.asizeof(self.memory))
            print(len(self.memory))

    def play(self, verbose:bool=False, sleep:float=0.1, max_steps:int=100):
        # Play an episode
        actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]

        iteration = 0
        state = self.env.reset()  # reset environment to a new, random state
        self.env.render()
        if verbose:
            print(f"Iter: {iteration} - Action: *** - Reward ***")
        time.sleep(sleep)
        done = False

        while not done: 
            action = self._get_action(state) 
            iteration += 1 
            state, reward, done, info = self.env.step(action) 
            #display.clear_output(wait=True) 
            self.env.render() 
            if verbose: 
                print(
                f"Iter: {iteration} - Action: {action}({actions_str[action]}) - Reward {reward}") 
            time.sleep(sleep) 
            if iteration >= max_steps: 
                print("cannot converge :(")
                break

if __name__ == "__main__":

    # Environment
    env = gym.make("Taxi-v3") # Step limit == 200
    #env = gym.make("Taxi-v3").env
    env.reset()
    #env.render()
