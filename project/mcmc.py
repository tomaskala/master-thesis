import abc
from typing import List

import numpy as np
from scipy.stats import rv_continuous

from utils import check_random_state


def update_truncnorm(loc, params):
    params['a'] = (params['a'] - loc) / params['scale']
    params['b'] = (params['b'] - loc) / params['scale']

    return params


class Distribution:
    def __init__(self, dist, truncnorm=False, **kwargs):
        assert 'loc' not in kwargs, 'the location parameter is to be given explicitly on each distribution invocation'

        self.dist = dist
        self.truncnorm = truncnorm
        self.kwargs = kwargs

        if truncnorm:
            if 'scale' not in kwargs or 'a' not in kwargs or 'b' not in kwargs:
                raise ValueError('Truncnorm not parameterized correctly.')

    def log_prob(self, x, loc):
        if self.truncnorm:
            params = update_truncnorm(loc, self.kwargs)
        else:
            params = self.kwargs

        return self.dist.logpdf(x=x, loc=loc, **params)

    def sample(self, loc, random_state=None):
        if self.truncnorm:
            params = update_truncnorm(loc, self.kwargs)
        else:
            params = self.kwargs

        return self.dist.rvs(loc=loc, random_state=random_state, **params)


class Prior:
    def __init__(self, distributions: List[rv_continuous]):
        self.distributions = distributions

    def log_prob(self, theta):
        assert len(self.distributions) == len(theta)
        log_prob = 0.0

        for th, dist in zip(theta, self.distributions):
            log_prob += dist.logpdf(th)

        return log_prob

    def sample(self, random_state=None):
        return np.array([dist.rvs(random_state=random_state) for dist in self.distributions], dtype=float)


class Proposal:
    def __init__(self, distributions: List[Distribution]):
        self.distributions = distributions

    def log_prob(self, theta, loc):
        assert len(self.distributions) == len(theta) == len(loc)
        log_prob = 0.0

        for th, l, dist in zip(theta, loc, self.distributions):
            log_prob += dist.log_prob(th, l)

        return log_prob

    def sample(self, loc, random_state=None):
        assert len(self.distributions) == len(loc)
        return np.array([dist.sample(l, random_state=random_state) for l, dist in zip(loc, self.distributions)],
                        dtype=float)


class MetropolisHastings(abc.ABC):
    def __init__(self,
                 n_samples,
                 prior,
                 proposal,
                 tune=True,
                 tune_interval=100,
                 theta_init=None,
                 random_state=None):
        self.n_samples = n_samples
        self.prior = prior
        self.proposal = proposal
        self.tune = tune
        self.tune_interval = tune_interval
        self.theta_init = theta_init
        self.random_state = check_random_state(random_state)

    def do_inference(self, y):
        """
        Infer the static parameters `theta` of the model from a sequence of 2-dimensional observations `y`.
        :param y: observation sequence: array, shaped (T, y-dim), T denotes the time
        :return: sampled parameters: array, shaped (n_samples, theta-dim)
        """
        theta = self.prior.sample() if self.theta_init is None else self.theta_init

        assert theta.ndim == 1
        assert y.ndim == 2

        thetas = np.zeros(shape=(self.n_samples, theta.shape[0]), dtype=float)
        loglik = -1e99
        accepted = 0

        for i in range(self.n_samples):
            theta_prop = self.proposal.sample(theta)
            log_ratio = 0.0

            log_ratio += self.prior.log_prob(theta_prop)
            log_ratio -= self.prior.log_prob(theta)

            log_ratio += self.proposal.log_prob(theta, theta_prop)
            log_ratio -= self.proposal.log_prob(theta_prop, theta)

            loglik_prop = self._log_likelihood_estimate(y, theta_prop)
            log_ratio += loglik_prop
            log_ratio -= loglik

            if np.log(self.random_state.rand()) < log_ratio:
                theta = theta_prop
                loglik = loglik_prop
                accepted += 1

            thetas[i] = theta

            print('Done sample {} of {} ({:.02f}%).'.format(i + 1, self.n_samples, (i + 1) / self.n_samples * 100.0))
            print('Accepted: {} of {} ({:.02f}%) samples so far.'.format(accepted, i + 1, accepted / (i + 1) * 100.0))

        return thetas

    @abc.abstractmethod
    def _log_likelihood_estimate(self, y, theta):
        pass


class MetropolisHastingsPF(MetropolisHastings, abc.ABC):
    def __init__(self,
                 n_samples,
                 n_particles,
                 state_init,
                 const,
                 prior,
                 proposal,
                 tune=True,
                 tune_interval=100,
                 theta_init=None,
                 random_state=None):
        super(MetropolisHastingsPF, self).__init__(n_samples=n_samples, prior=prior, proposal=proposal, tune=tune,
                                                   tune_interval=tune_interval, theta_init=theta_init,
                                                   random_state=random_state)

        self.n_particles = n_particles
        self.state_init = state_init
        self.const = const

    def _log_likelihood_estimate(self, y, theta):
        if self.state_init.ndim == 1:
            x = np.tile(self.state_init, (self.n_particles, 1))
        else:
            x = self.state_init

        assert x.ndim == 2 and x.shape[0] == self.n_particles
        T = y.shape[0]
        x_dim = x.shape[1]
        loglik = 0.0

        for t in range(T):
            x = self._transition(x, t + 1, theta)
            assert x.ndim == 2 and x.shape == (self.n_particles, x_dim)

            lw = self._observation_log_prob(y[t], x, theta)
            w = np.exp(lw)
            assert lw.ndim == 1 and lw.shape[0] == self.n_particles

            if np.max(w) < 1e-20:
                print('Warning: Particle filter bombed.')
                return -1e99

            loglik += np.log(np.mean(w))
            rows = self.random_state.choice(self.n_particles, self.n_particles, replace=True, p=w / np.sum(w))
            x = x[rows]

        return loglik

    @abc.abstractmethod
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _observation_log_prob(self, y: np.ndarray, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        pass
