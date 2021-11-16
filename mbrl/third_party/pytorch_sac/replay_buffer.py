import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = torch.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype).to(device)
        self.next_obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype).to(device)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32).to(device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32).to(device)
        self.not_dones = torch.empty((capacity, 1), dtype=torch.float32).to(device)
        self.not_dones_no_max = torch.empty((capacity, 1), dtype=torch.float32).to(device)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, from_np=True):
        if from_np:
            obs = torch.from_numpy(obs).float().to(self.device)
            action = torch.from_numpy(action).float().to(self.device)
            reward = torch.from_numpy(np.array(reward)).float().to(self.device)
            next_obs = torch.from_numpy(next_obs).float().to(self.device)
            done = torch.from_numpy(np.array(done)).float().to(self.device)
            done_no_max = torch.from_numpy(np.array(done_no_max)).float().to(self.device)

        self.obses[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obses[self.idx] = next_obs
        self.not_dones[self.idx] = torch.logical_not(done)
        self.not_dones_no_max[self.idx] = torch.logical_not(done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_batch(self, obs, action, reward, next_obs, done, done_no_max, from_np=True):
        if from_np:
            obs = torch.from_numpy(obs).float().to(self.device)
            action = torch.from_numpy(action).float().to(self.device)
            reward = torch.from_numpy(reward).float().to(self.device)
            next_obs = torch.from_numpy(next_obs).float().to(self.device)
            done = torch.from_numpy(done).float().to(self.device)
            done_no_max = torch.from_numpy(done_no_max).float().to(self.device)

        def copy_from_to(buffer_start, batch_start, how_many):
            buffer_slice = slice(buffer_start, buffer_start + how_many)
            batch_slice = slice(batch_start, batch_start + how_many)
            self.obses[buffer_slice] = obs[batch_slice]
            self.actions[buffer_slice] = action[batch_slice]
            self.rewards[buffer_slice] = reward[batch_slice]
            self.next_obses[buffer_slice] = next_obs[batch_slice]
            self.not_dones[buffer_slice] = torch.logical_not(done[batch_slice])
            self.not_dones_no_max[buffer_slice] = torch.logical_not(done_no_max[batch_slice])


        _batch_start = 0
        buffer_end = self.idx + len(obs)
        if buffer_end > self.capacity:
            copy_from_to(self.idx, _batch_start, self.capacity - self.idx)
            _batch_start = self.capacity - self.idx
            self.idx = 0
            self.full = True

        _how_many = len(obs) - _batch_start
        copy_from_to(self.idx, _batch_start, _how_many)
        self.idx = (self.idx + _how_many) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, sac=True):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = self.obses[idxs].float()
        actions = self.actions[idxs].float()
        rewards = self.rewards[idxs].float()
        next_obses = self.next_obses[idxs].float()
        not_dones = self.not_dones[idxs].float()
        not_dones_no_max = self.not_dones_no_max[idxs].float()

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def save(self, path):
        torch.save(self.obses, path + "/buffer_obses.torch")
        torch.save(self.actions, path + "/buffer_actions.torch")
        torch.save(self.rewards, path + "/buffer_rewards.torch")
        torch.save(self.next_obses, path + "/buffer_next_obses.torch")
        torch.save(self.not_dones, path + "/buffer_not_dones.torch")
        torch.save(self.not_dones_no_max, path + "/buffer_not_dones_no_max.torch")

        torch.save(self.idx, path + "/buffer_idx.torch")
        torch.save(self.last_save, path + "/buffer_last_save.torch")
        torch.save(self.full, path + "/buffer_full.torch")
        torch.save(self.capacity, path + "/buffer_capacity.torch")

    def load(self, path):
        self.obses = torch.load(path + "/buffer_obses.torch")
        self.actions = torch.load(path + "/buffer_actions.torch")
        self.rewards = torch.load(path + "/buffer_rewards.torch")
        self.next_obses = torch.load(path + "/buffer_next_obses.torch")
        self.not_dones = torch.load(path + "/buffer_not_dones.torch")
        self.not_dones_no_max = torch.load(path + "/buffer_not_dones_no_max.torch")

        self.idx = torch.load(path + "/buffer_idx.torch")
        self.last_save = torch.load(path + "/buffer_last_save.torch")
        self.full = torch.load(path + "/buffer_full.torch")
        self.capacity = torch.load(path + "/buffer_capacity.torch")

