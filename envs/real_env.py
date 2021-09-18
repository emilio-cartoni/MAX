import real_robots
import gym
import numpy as np

def get_state_block(state):
    x = state[0].item()
    y = state[1].item()
    block_x = int(np.floor((min(max(y, -0.4), 0.2) - (-0.4)) / 0.6 * 100))
    block_y = int(np.floor((min(max(y, -0.5), 0.5) - (-0.5)) / 1.0 * 100))
    return block_x * 100 + block_y

def rate_buffer(buffer):
    visited_blocks = [get_state_block(state) for state in buffer.states]
    n_unique = len(set(visited_blocks))
    return n_unique
    
class MyEnv(real_robots.envs.env.REALRobotEnv):

    def __init__(self):
        super().__init__(self, objects=1, action_type='macro_action')
        
        #self.observation_space = gym.spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        high = np.array([np.finfo(np.float32).max,
                 np.finfo(np.float32).max,
                 np.finfo(np.float32).max])
        self.observation_space = gym.spaces.Box(-high, high, dtype=float)
        
        self.action_space = gym.spaces.Box(
                                  low=np.array([-0.25, -0.5, -0.25, -0.5]),
                                  high=np.array([0.05, 0.5, 0.05, 0.5]),
                                  dtype=float)
        
        self.step = self.my_step

    def reset(self):
        obs = super().reset()
        return obs['object_positions']['cube']

    def my_step(self, action):
        m_action = {"macro_action": action.reshape(2, 2), "render": False}
        for _ in range(999):
            self.step_macro(m_action)
        observation, reward, done, info = self.step_macro(m_action)
        return observation['object_positions']['cube'], reward, done, info


