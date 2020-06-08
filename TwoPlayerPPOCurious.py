### Something wrong with what I did with conv here?

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym
from twoSnakeGameGymCurious import TwoSnakeGameGym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## Initializaes weights to be a "good" random
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        ## "Good" initialization
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        ### Sequential
    
        if type(state_dim) == int:
            # actor
            self.action_layer = nn.Sequential(
                    init_(nn.Linear(state_dim, n_latent_var)),
                    nn.ReLU(),
                    init_(nn.Linear(n_latent_var, n_latent_var)),
                    nn.ReLU(),
                    init_(nn.Linear(n_latent_var, action_dim)),
                    nn.Softmax(dim=-1)
                    )
            # critic
            self.value_layer = nn.Sequential(
                    init_(nn.Linear(state_dim, n_latent_var)),
                    nn.ReLU(),
                    init_(nn.Linear(n_latent_var, n_latent_var)),
                    nn.ReLU(),
                    init_(nn.Linear(n_latent_var, 1))
                    )
        else:
            self.action_layer = nn.Sequential(
                    init_(nn.Conv2d(1, 5, 3, stride=1)), nn.ReLU(),
                    init_(nn.Conv2d(5, 2, 3, stride=1)), nn.ReLU(), Flatten(),
                    init_(nn.Linear(408, action_dim)), nn.ReLU(),
                    nn.Softmax(dim=-1)
                    )
            # critic
            self.value_layer = nn.Sequential(
                    init_(nn.Conv2d(1, 5, 3, stride=1)), nn.ReLU(),
                    init_(nn.Conv2d(5, 2, 3, stride=1)), nn.ReLU(), Flatten(),
                    init_(nn.Linear(408, n_latent_var)), nn.ReLU(),
                    nn.Linear(n_latent_var, 1)
                )

        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state = torch.from_numpy(state).double()
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    torch.set_default_tensor_type('torch.DoubleTensor')
    ############## Hyperparameters ##############
    #env_name = "LunarLander-v2"
    # creating environment
    #env = gym.make(env_name)

    env_name = "Two_snake_game"
    env = TwoSnakeGameGym()

    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 5000         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 1000       # max training episodes
    max_timesteps = 1000         # max timesteps in one episode
    n_latent_var = 40           # number of variables in hidden layer
    update_timestep = 600      # update policy every n timesteps
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 1.00                # discount factor -- we want 0 because duh 
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory_one = Memory()
    ppo_one = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    memory_two = Memory()
    ppo_two = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)


    use_last_prog = False
    if use_last_prog:
        filename_one = "PPO_one_{}.pth".format(env_name)
        filename_two = "PPO_two_{}.pth".format(env_name)
        directory = ""
        ppo_one.policy.load_state_dict(torch.load(directory+filename_one))
        ppo_two.policy.load_state_dict(torch.load(directory+filename_two))

    print(lr,betas)
    
    # logging variables
    running_reward_one = 0
    running_reward_two = 0

    running_scores_one = 0
    running_scores_two = 0

    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state_one, state_two = env.reset()

        for t in range(max_timesteps):
            timestep += 1
            # To players, they take turns simultaneously 

            # Running policy_old:
            action_one = ppo_one.policy_old.act(state_one, memory_one)
            state_one, reward_one, done_one, scores_one = env.step(action_one, 0)

            action_two = ppo_two.policy_old.act(state_two, memory_two)
            state_two, reward_two, done_two, scores_two = env.step(action_two, 1)

            done = done_one or done_two
            
            # Saving reward and is_terminal:
            memory_one.rewards.append(reward_one)
            memory_one.is_terminals.append(done)

            memory_two.rewards.append(reward_two)
            memory_two.is_terminals.append(done)
            
            # update if it's time
            if timestep % update_timestep == 0:
                ppo_one.update(memory_one)
                ppo_two.update(memory_two)

                memory_one.clear_memory()
                memory_two.clear_memory()
                timestep = 0

            
            running_reward_one += reward_one
            running_reward_two += reward_two

            running_scores_one += scores_one["episode"]["r"]
            running_scores_two += scores_two["episode"]["r"]

            if render:
                env.render()
            if done:
                break
                
        avg_length += t
            
        # stop training if avg_reward > solved_reward
        if running_reward_one > (log_interval*solved_reward):
            print("########## Solved! ##########")
            avg_length = int(avg_length/log_interval)
            running_reward_one = int((running_reward_one/log_interval))
            running_reward_two = int((running_reward_two/log_interval))
            running_scores_one = int((running_scores_one/log_interval))
            running_scores_two = int((running_scores_two/log_interval))
            print('Episode {} \t avg length: {} \t reward_one: {}, scores_one: {}'.format(i_episode, avg_length, running_reward_one, running_scores_one))
            print('Episode {} \t avg length: {} \t reward_two: {}, scores_two: {}'.format(i_episode, avg_length, running_reward_two, running_scores_two))

            running_scores_one = 0
            running_scores_two = 0
            running_reward_one = 0
            running_reward_two = 0 

            torch.save(ppo_one.policy.state_dict(), './PPO_curious_one_{}.pth'.format(env_name))
            torch.save(ppo_one.policy.state_dict(), './PPO_curious_two_{}.pth'.format(env_name))
            break
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward_one = int((running_reward_one/log_interval))
            running_reward_two = int((running_reward_two/log_interval))
            running_scores_one = int((running_scores_one/log_interval))
            running_scores_two = int((running_scores_two/log_interval))
            print('Episode {} \t avg length: {} \t reward_one: {}, scores_one: {}'.format(i_episode, avg_length, running_reward_one, running_scores_one))
            print('Episode {} \t avg length: {} \t reward_two: {}, scores_two: {}'.format(i_episode, avg_length, running_reward_two, running_scores_two))

            running_scores_one = 0
            running_scores_two = 0
            running_reward_one = 0
            running_reward_two = 0 
            avg_length = 0
        ## after we hit max episodes
        ##torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
    
    torch.save(ppo_one.policy.state_dict(), './PPO_curious_one_progress{}.pth'.format(env_name))
    torch.save(ppo_one.policy.state_dict(), './PPO_curious_two_progress{}.pth'.format(env_name))
if __name__ == '__main__':
    main()
    
