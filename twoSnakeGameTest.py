import gym
from TwoPlayerPPO import PPO, Memory
from PIL import Image
import torch
from snakeGameGym import SnakeGameGym
from twoSnakeGameGym import TwoSnakeGameGym
import time


def test():
    torch.set_default_tensor_type('torch.DoubleTensor')
    ############## Hyperparameters ##############
    env_name = "Two_snake_game"
    env = TwoSnakeGameGym(initBoard=True)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    max_timesteps = 500
    n_latent_var = 64         # number of variables in hidden layer
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 1.00                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    n_episodes = 3
    max_timesteps = 300
    render = True
    save_gif = False

    filename_one = "PPOone_{}.pth".format(env_name)
    filename_two = "PPOtwo_{}.pth".format(env_name)
    directory = ""
    

    memory_one = Memory()
    ppo_one = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    memory_two = Memory()
    ppo_two = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    
    ppo_one.policy.load_state_dict(torch.load(directory+filename_one))
    ppo_two.policy.load_state_dict(torch.load(directory+filename_two))
    
    for ep in range(1, n_episodes+1):
        ep_reward_one = 0
        ep_reward_two = 0
        state_one, state_two = env.reset()

        for t in range(max_timesteps):
            print(t)
            ## So that it doesn't go too fast and goes at a normal snake game pace
            time.sleep(0.05);
             # Running policy_old:
            action_one = ppo_one.policy.act(state_one, memory_one)
            state_one, reward_one, done_one, _ = env.step(action_one, 0)

            action_two = ppo_two.policy.act(state_two, memory_two)
            state_two, reward_two, done_two, _ = env.step(action_two, 1)

            ep_reward_one += reward_one
            ep_reward_two += reward_two

            done = done_one or done_two

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
            
        print('Episode: {}\tReward_One: {}'.format(ep, int(ep_reward_one)))
        print('Episode: {}\tReward_Two: {}'.format(ep, int(ep_reward_two)))
        ep_reward_one = 0
        ep_reward_two = 0
        env.close()
    
if __name__ == '__main__':
    test()
    
    
