import gym
from PPO import PPO, Memory
from PIL import Image
import torch
from snakeGameGym import SnakeGameGym
import time


def test():
    torch.set_default_tensor_type('torch.DoubleTensor')
    ############## Hyperparameters ##############
    env_name = "Snake Game"
    # creating environment
    # tell env to initialize game board too 
    env = SnakeGameGym(initBoard=True)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    max_timesteps = 500
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 1.00                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    n_episodes = 3
    max_timesteps = 300
    render = True
    save_gif = False

    filename = "PPO_{}.pth".format(env_name)
    directory = ""
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    ppo.policy.load_state_dict(torch.load(directory+filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(max_timesteps):
            print(t)
            ## So that it doesn't go too fast and goes at a normal snake game pace
            time.sleep(0.05);
            action = ppo.policy.act(state, memory)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                print("Rendering")
                env.render()
            #if save_gif:
            #    img = env.render(mode = 'rgb_array')
            #    img = Image.fromarray(img)
            #     img.save('./gif/{}.jpg'.format(t))  
            if done:
                print("Done")
                env.cleanup()
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()
    
if __name__ == '__main__':
    test()
    
    
