import abc
import six
import numpy as np
import numpy.random as nr
from overrides import overrides


@six.add_metaclass(abc.ABCMeta)
class ExplorationStrategy():

    @abc.abstractmethod
    def noise_action(self, action, t, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass


class OUNoise(ExplorationStrategy):
    """docstring for OUNoise"""

    def __init__(self, env, mu=0, theta=0.15, sigma=0.3):
        self.env = env
        self.action_dimension = self.env.action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    @overrides
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def add_noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    def noise_action(self, action, t=1, **kwargs):
        ou_state = self.add_noise()
        return np.clip(action + ou_state, self.env.action_low, self.env.action_high)


class GaussianStrategy(ExplorationStrategy):
    """
        This strategy adds Gaussian noise to the action taken by the deterministic policy.
    """

    def __init__(self, env, max_sigma=1.0, min_sigma=0.1, decay_period=1000000):
        assert len(env.action_space.shape) == 1
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = env.action_space

    def noise_action(self, action, t, **kwargs):
        sigma = self._max_sigma - \
            (self._max_sigma - self._min_sigma) * \
            min(1.0, t * 1.0) / self._decay_period
        return np.clip(action + np.random.normal(size=len(action)) * sigma, self.env.action_low, self.env.action_high)
