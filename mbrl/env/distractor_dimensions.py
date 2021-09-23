import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box


class Distractor(ObservationWrapper):

    def __init__(self, env, dimensions=10, linear=True, switching=False, correlated=True):
        super().__init__(env)
        self.dimensions = dimensions
        self.linear = linear
        self.switching = switching

        obs_space = env.observation_space
        low = np.concatenate((obs_space.low, np.array([-1 * np.inf] * dimensions)), -1)
        high = np.concatenate((obs_space.high, np.array([np.inf] * dimensions)), -1)
        self.observation_space = Box(low, high)

        self.distractor_state = np.random.normal(0, 0.1, size=(dimensions,))
        self.linear_map = np.random.normal(0, 1., size=(dimensions, high.shape[0]))
        if not correlated:
            self.linear_map[:, :obs_space.shape[0]] = 0.

        self.random_sin_parameters = np.random.normal(0, 10., size = (1, dimensions, dimensions)) ** 2
        self.reset_switching = np.random.normal(size=(dimensions,))

    def observation(self, obs):
        self.distractor_state = np.matmul(self.linear_map, np.concatenate((obs, self.distractor_state)))

        if not self.linear:
            self.distractor_state += 0.1 * np.sum(np.sin(np.matmul(self.random_sin_parameters, self.distractor_state)), 0)
        
        if self.switching:
            self.distractor_state = np.where(np.abs(self.distractor_state) > 20., self.reset_switching, self.distractor_state)

        return np.concatenate((obs, self.distractor_state), 0).astype(np.float)

    def reset(self):
        obs = self.env.reset()
        self.distractor_state = np.random.normal(0, 0.1, size=(self.dimensions,))
        return self.observation(obs)
