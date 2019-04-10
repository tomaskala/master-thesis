import abc
import warnings
from typing import List

import numpy as np
from scipy.stats import cauchy, norm, rv_continuous

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
        theta = self.prior.sample(random_state=self.random_state) if self.theta_init is None else self.theta_init

        assert theta.ndim == 1
        assert y.ndim == 2

        thetas = np.zeros(shape=(self.n_samples, theta.shape[0]), dtype=float)
        loglik = -1e99
        accepted = 0

        for i in range(self.n_samples):
            theta_prop = self.proposal.sample(theta, random_state=self.random_state)
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
            assert lw.ndim == 1 and lw.shape == (self.n_particles,)
            w = np.exp(lw)

            if np.max(w) < 1e-20:
                warnings.warn('Weight underflow.', RuntimeWarning)
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


class Kernel(abc.ABC):
    """
    Base class for all kernel functions -- symmetric distributions in the location-scale family.
    The kernels support two operations:
    1. log_kernel
       Given an array of N pseudo-measurements (N being the number of particles) and a single true
       measurement y, calculate an array of N log-probabilities of each pseudo-measurement with the
       kernel centered at y.
    2. tune_scale
       Given the alpha-th closest pseudo-measurement u to the true measurement y, and the high probability
       region (HPR) probability p, calculate the kernel scale so that it covers alpha/N with a p-HPR, once
       centered at y.
    """

    def __init__(self):
        self.scale = 1.0
        self.scale_log = [self.scale]

    @abc.abstractmethod
    def log_kernel(self, u: np.ndarray, y: float) -> np.ndarray:
        """
        Evaluate the logarithm of the kernel on an array of pseudo-observations and a true observation.
        :param u: array, shape (n_particles,)
        :param y: scalar
        :return: array, shape (n_particles,)
        """
        pass

    def tune_scale(self, u: float, y: float, p: float):
        self._tune_scale(u, y, p)
        self.scale_log.append(self.scale)

    @abc.abstractmethod
    def _tune_scale(self, u: float, y: float, p: float):
        pass

    @abc.abstractmethod
    def sample(self, size=None, random_state=None):
        pass


class GaussianKernel(Kernel):
    def log_kernel(self, u: np.ndarray, y: float) -> np.ndarray:
        return -np.power(u - y, 2.0) / (2.0 * (self.scale ** 2.0))

    def _tune_scale(self, u: float, y: float, p: float):
        self.scale = np.abs(u - y) / norm.ppf(q=((p + 1) / 2))

    def sample(self, size=None, random_state=None):
        return norm.rvs(loc=0.0, scale=self.scale, size=size, random_state=random_state)


class CauchyKernel(Kernel):
    def log_kernel(self, u: np.ndarray, y: float) -> np.ndarray:
        return -np.log1p(np.power(u - y, 2.0) / (self.scale ** 2.0))

    def _tune_scale(self, u: float, y: float, p: float):
        self.scale = np.abs(u - y) / cauchy.ppf(q=((p + 1) / 2))

    def sample(self, size=None, random_state=None):
        return cauchy.rvs(loc=0.0, scale=self.scale, size=size, random_state=random_state)


class MetropolisHastingsABC(MetropolisHastings, abc.ABC):
    def __init__(self,
                 n_samples,
                 n_particles,
                 alpha,
                 hpr_p,
                 state_init,
                 const,
                 kernel,
                 prior,
                 proposal,
                 tune=True,
                 tune_interval=100,
                 theta_init=None,
                 random_state=None):
        assert int(alpha * n_particles) <= n_particles, \
            'the number of covered pseudo-measurements must be at most the number of particles'

        super(MetropolisHastingsABC, self).__init__(n_samples=n_samples, prior=prior, proposal=proposal, tune=tune,
                                                    tune_interval=tune_interval, theta_init=theta_init,
                                                    random_state=random_state)

        self.n_particles = n_particles
        self.n_particles_covered = int(alpha * n_particles)
        self.hpr_p = hpr_p
        self.state_init = state_init
        self.const = const

        if kernel == 'gaussian':
            self.kernel = GaussianKernel()
            # self.kernel_f = GaussianKernel
        elif kernel == 'cauchy':
            self.kernel = CauchyKernel()
            # self.kernel_f = CauchyKernel
        else:
            raise ValueError('Unknown kernel: {}.'.format(kernel))

        # self.kernel = []

    # def do_inference(self, y):
    #     # Create a separate kernel for each y dimension.
    #     self.kernel = [self.kernel_f() for _ in range(y.shape[0])]
    #     return super(MetropolisHastingsABC, self).do_inference(y=y)

    def _log_likelihood_estimate(self, y, theta):
        if self.state_init.ndim == 1:
            x = np.tile(self.state_init, (self.n_particles, 1))
        else:
            x = self.state_init

        assert x.ndim == 2 and x.shape[0] == self.n_particles
        # T, y_dim = y.shape
        T = y.shape[0]
        x_dim = x.shape[1]
        loglik = 0.0

        for t in range(T):
            x = self._transition(x, t + 1, theta)
            assert x.ndim == 2 and x.shape == (self.n_particles, x_dim)

            u = self._measurement_model(x, theta)
            # assert u.ndim == 2 and u.shape == (self.n_particles, )

            u_alpha = self._alphath_closest(u=u, y_t=y[t])
            # assert u_alpha.ndim == 1 and u_alpha.shape == (y_dim,)

            self._tune_kernel_scales(u_alpha=u_alpha, y_t=y[t])

            lw = self._log_kernel(u=u, y_t=y[t])
            assert lw.ndim == 1 and lw.shape == (self.n_particles,)
            w = np.exp(lw)

            if np.max(w) < 1e-20:
                warnings.warn('Weight underflow.', RuntimeWarning)
                return -1e99

            loglik += np.log(np.mean(w))
            rows = self.random_state.choice(self.n_particles, self.n_particles, replace=True, p=w / np.sum(w))
            x = x[rows]

        return loglik

    def _alphath_closest(self, u: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        distances_squared = np.power(y_t - u, 2)
        alphath_closest_idx = np.argpartition(distances_squared, kth=self.n_particles_covered - 1)[
            self.n_particles_covered - 1]
        return u[alphath_closest_idx]

        # y_dim = y_t.shape[0]
        # assert u.ndim == 2 and u.shape == (self.n_particles, y_dim)
        # assert y_t.ndim == 1 and y_t.shape == (y_dim,)
        #
        # distances_squared = np.power(y_t[np.newaxis, :] - u, 2)
        #
        # # Alpha denotes the number of pseudo-measurements covered by the p-HPR of the kernel. However,
        # # indexing is 0-based, so we subtract 1 to get the alphath closest pseudo-measurement to y.
        # alphath_closest_idx = np.argpartition(distances_squared, kth=self.n_particles_covered - 1, axis=0)[
        #     self.n_particles_covered - 1]
        # assert alphath_closest_idx.ndim == 1 and alphath_closest_idx.shape == (y_dim,)
        #
        # return u[alphath_closest_idx, np.arange(y_dim)]

    def _tune_kernel_scales(self, u_alpha: np.ndarray, y_t: np.ndarray):
        # assert u_alpha.ndim == 1 and u_alpha.shape == y_t.shape
        #
        # for u_elem, y_elem, kernel in zip(u_alpha, y_t, self.kernel):
        #     kernel.tune_scale(u=u_elem, y=y_elem, p=self.hpr_p)
        self.kernel.tune_scale(u_alpha, y_t, self.hpr_p)

    def _sample_from_kernel(self) -> np.ndarray:  # TODO: Use if noisy_abc is True.
        # return np.array([kernel.sample(random_state=self.random_state) for kernel in self.kernel], dtype=float)
        return self.kernel.sample(random_state=self.random_state)

    def _log_kernel(self, u: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        # y_dim = y_t.shape[0]
        # assert u.ndim == 2 and u.shape == (self.n_particles, y_dim)
        # assert y_t.ndim == 1 and y_t.shape == (y_dim,)
        #
        # log_kernel = np.zeros(shape=self.n_particles, dtype=float)
        #
        # for u_elem, y_elem, kernel in zip(u.T, y_t, self.kernel):
        #     lk = kernel.log_kernel(u=u_elem, y=y_elem)
        #     assert lk.ndim == 1 and lk.shape == (self.n_particles,)
        #     log_kernel += lk
        #
        # return log_kernel
        return self.kernel.log_kernel(u, y_t)

    @abc.abstractmethod
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _measurement_model(self, x: np.ndarray, theta: np.ndarray) -> np.array:
        pass
