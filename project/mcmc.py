import abc
import math
import numbers
from typing import List, Union

import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import cauchy, laplace, norm, rv_continuous, uniform

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
            params = update_truncnorm(loc, self.kwargs.copy())
        else:
            params = self.kwargs

        return self.dist.logpdf(x=x, loc=loc, **params)

    def sample(self, loc, random_state=None):
        if self.truncnorm:
            params = update_truncnorm(loc, self.kwargs.copy())
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
        self.scaling = 1.0
        self.steps_until_tune = tune_interval
        self.accepted = 0
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
        # assert y.ndim == 2

        thetas = np.zeros(shape=(self.n_samples, theta.shape[0]), dtype=float)
        loglik = -1e99
        accepted = 0

        for i in range(self.n_samples):
            # if not self.steps_until_tune and self.tune: todo
            #     self._tune_scale()
            #     self.steps_until_tune = self.tune_interval
            #     self.accepted = 0

            theta_prop = self.proposal.sample(theta, random_state=self.random_state)

            # if self.tune: todo
            #     theta_prop = theta + (theta_prop - theta) * self.scaling

            log_ratio = 0.0

            log_ratio += self.prior.log_prob(theta_prop)
            log_ratio -= self.prior.log_prob(theta)

            log_ratio += self.proposal.log_prob(theta, theta_prop)
            log_ratio -= self.proposal.log_prob(theta_prop, theta)

            loglik_prop = self._log_likelihood_estimate(y, theta_prop)
            log_ratio += loglik_prop
            log_ratio -= loglik

            if math.log(self.random_state.rand()) < log_ratio:
                theta = theta_prop
                loglik = loglik_prop
                accepted += 1
                # self.accepted += 1 todo

            thetas[i] = theta
            print('THETA', theta)
            # self.steps_until_tune -= 1 todo

            print('Done sample {} of {} ({:.02f}%).'.format(i + 1, self.n_samples, (i + 1) / self.n_samples * 100.0))
            print('Accepted: {} of {} ({:.02f}%) samples so far.'.format(accepted, i + 1, accepted / (i + 1) * 100.0))

        return thetas

    def _tune_scale(self):
        """
        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate over the last tune_interval:
         Rate   Variance adaptation
        ------  -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2.0
        >0.95         x 10.0
        Shamelessly stolen from:
        `https://github.com/pymc-devs/pymc3/blob/master/pymc3/step_methods/metropolis.py`.
        """
        acceptance_rate = self.accepted / self.tune_interval

        if acceptance_rate < 0.001:
            # Reduce by 90 percent.
            self.scaling *= 0.1
        elif acceptance_rate < 0.05:
            # Reduce by 50 percent.
            self.scaling *= 0.5
        elif acceptance_rate < 0.2:
            # Reduce by 10 percent.
            self.scaling *= 0.9
        elif acceptance_rate > 0.95:
            # Increase by a factor of 10.
            self.scaling *= 10.0
        elif acceptance_rate > 0.75:
            # Increase by a factor of 2.
            self.scaling *= 2.0
        elif acceptance_rate > 0.5:
            # Increase by 10 percent.
            self.scaling *= 1.1

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
            loglik += logsumexp(lw)

            rows = self.random_state.choice(self.n_particles, self.n_particles, replace=True, p=softmax(lw))
            x = x[rows]

        loglik -= T * math.log(self.n_particles)
        assert loglik <= 0.0
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

    def __init__(self, p: float):
        self.p = p
        self.scale = 1.0
        # self.scale_log = [self.scale]

    @abc.abstractmethod
    def __call__(self, u: np.ndarray, y: float) -> np.ndarray:
        """
        Evaluate the log-kernel on an array of pseudo-observations and a true observation.
        :param u: array, shape (n_particles,)
        :param y: scalar
        :return: array, shape (n_particles,)
        """
        pass

    def tune_scale(self, u: float, y: float):
        self._tune_scale(u=u, y=y)
        # self.scale_log.append(self.scale)

    @abc.abstractmethod
    def _tune_scale(self, u: float, y: float):
        pass


class GaussianKernel(Kernel):
    def __call__(self, u: np.ndarray, y: float) -> np.ndarray:
        return norm.logpdf(x=u, loc=y, scale=self.scale)

    def _tune_scale(self, u: float, y: float):
        self.scale = abs(u - y) / norm.ppf(q=((self.p + 1) / 2))


class CauchyKernel(Kernel):
    def __call__(self, u: np.ndarray, y: float) -> np.ndarray:
        return cauchy.logpdf(x=u, loc=y, scale=self.scale)

    def _tune_scale(self, u: float, y: float):
        self.scale = abs(u - y) / cauchy.ppf(q=((self.p + 1) / 2))


class LaplaceKernel(Kernel):
    def __call__(self, u: np.ndarray, y: float) -> np.ndarray:
        return laplace.logpdf(x=u, loc=y, scale=self.scale)

    def _tune_scale(self, u: float, y: float):
        self.scale = abs(u - y) / laplace.ppf(q=((self.p + 1) / 2))


class UniformKernel(Kernel):
    def __call__(self, u: np.ndarray, y: float) -> np.ndarray:
        loc = y - self.scale / 2
        return uniform.logpdf(x=u, loc=loc, scale=self.scale)

    def _tune_scale(self, u: float, y: float):
        self.scale = abs(u - y) / uniform.ppf(q=((self.p + 1) / 2))


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
        self.kernel = None

        if kernel == 'gaussian':
            self.kernel_f = GaussianKernel
        elif kernel == 'cauchy':
            self.kernel_f = CauchyKernel
        elif kernel == 'laplace':
            self.kernel_f = LaplaceKernel
        elif kernel == 'uniform':
            self.kernel_f = UniformKernel
        else:
            raise ValueError('Unknown kernel: {}.'.format(kernel))

    def do_inference(self, y):
        if y.ndim == 1:
            self.kernel = self.kernel_f(self.hpr_p)
        else:
            self.kernel = [self.kernel_f(self.hpr_p) for _ in range(y.shape[1])]

        return super(MetropolisHastingsABC, self).do_inference(y=y)

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

            u = self._measurement_model(x, theta)
            assert (u.ndim == 1 and u.shape == (self.n_particles,)) or (
                    u.ndim == 2 and u.shape == (self.n_particles, y.shape[1]))

            u_alpha = self._alphath_closest(u=u, y_t=y[t])
            assert (y.ndim == 1 and isinstance(u_alpha, numbers.Number)) or (
                    y.ndim == 2 and u_alpha.shape == (y.shape[1],))
            self._tune_kernel_scales(u_alpha=u_alpha, y_t=y[t])

            lw = self._evaluate_kernel(u=u, y_t=y[t])
            assert lw.ndim == 1 and lw.shape == (self.n_particles,)
            loglik += logsumexp(lw)

            rows = self.random_state.choice(self.n_particles, self.n_particles, replace=True, p=softmax(lw))
            x = x[rows]

        loglik -= T * math.log(self.n_particles)
        assert loglik <= 0.0
        return loglik

    def _alphath_closest(self, u: np.ndarray, y_t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Alpha denotes the number of pseudo-measurements covered by the p-HPR of the kernel. However,
        # indexing is 0-based, so we subtract 1 to get the alphath closest pseudo-measurement to y.
        if isinstance(self.kernel, Kernel):
            # Univariate observations -> single kernel.
            dists = np.abs(u - y_t)
            alphath_closest_idx = np.argpartition(dists, kth=self.n_particles_covered - 1, axis=-1)[
                self.n_particles_covered - 1]
            return u[alphath_closest_idx]
        else:
            # Multivariate observations -> one kernel per y-dimension.
            dists = np.abs(u - y_t[np.newaxis, :])
            alphath_closest_idx = np.argpartition(dists, kth=self.n_particles_covered - 1, axis=0)[
                self.n_particles_covered - 1]

            y_dim = y_t.shape[0]
            assert alphath_closest_idx.shape == (y_dim,)
            return u[alphath_closest_idx, np.arange(y_dim)]

    def _tune_kernel_scales(self, u_alpha, y_t):
        if isinstance(self.kernel, Kernel):
            # Univariate observations -> single kernel.
            self.kernel.tune_scale(u=u_alpha, y=y_t)
        else:
            # Multivariate observations -> one kernel per y-dimension.
            for i, kernel in enumerate(self.kernel):
                kernel.tune_scale(u=u_alpha[i], y=y_t[i])

    def _evaluate_kernel(self, u, y_t):
        if isinstance(self.kernel, Kernel):
            # Univariate observations -> single kernel.
            return self.kernel(u=u, y=y_t)
        else:
            # Multivariate observations -> one kernel per y-dimension. Assume independent y-dimensions.
            lws = np.array([kernel(u=u[:, i], y=y_t[i]) for i, kernel in enumerate(self.kernel)])
            return np.sum(lws, axis=0)

    @abc.abstractmethod
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _measurement_model(self, x: np.ndarray, theta: np.ndarray) -> np.array:
        pass
