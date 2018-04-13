import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class TabularMDPEnv(gym.Env):
    def __init__(self, num_states, num_actions):
        super(TabularMDPEnv, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=0.0,
            high=1.0, shape=(num_states,), dtype=np.float32)

        self._task = None
        self._transitions = None
        self._rewards_mean = None
        self._state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        transitions = self.np_random.dirichlet(np.ones(self.num_states),
            size=(num_tasks, self.num_states, self.num_actions))
        rewards_mean = self.np_random.normal(1.0, 1.0,
            size=(num_tasks, self.num_states, self.num_actions))
        tasks = [{'transitions': transition, 'rewards_mean': reward_mean}
            for (transition, reward_mean) in zip(transitions, rewards_mean)]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._transitions = task['transitions']
        self._rewards_mean = task['rewards_mean']

    def reset(self):
        # From Duan 2016: "an episode always starts on the first state"
        self._state = 0
        observation = np.zeros(self.num_states, dtype=np.float32)
        observation[self._state] = 1.0

        return observation

    def step(self, action):
        assert self.action_space.contains(action)
        mean = self._rewards_mean[self._state, action]
        reward = self.np_random.normal(mean, 1.0)

        self._state = self.np_random.choice(self.num_states,
            p=self._transitions[self._state, action])
        observation = np.zeros(self.num_states, dtype=np.float32)
        observation[self._state] = 1.0

        return observation, reward, False, self._task
