import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import logsumexp
from scipy.stats import cauchy, norm, rv_continuous

from utils import check_random_state


# TODO: Multivariate measurements => each coordinate can have its own kernel function.
# TODO: If we assume independence, we can just multiply kernels === sum log-kernels.
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
    def log_kernel(self, u: np.ndarray, center: float) -> np.ndarray:
        pass

    def tune_scale(self, y: float, u: float, p: float):
        self._tune_scale(y=y, u=u, p=p)
        self.scale_log.append(self.scale)

    @abc.abstractmethod
    def _tune_scale(self, y: float, u: float, p: float):
        pass

    @abc.abstractmethod
    def sample(self, size: Optional[int] = None, random_state=None) -> Union[float, np.ndarray]:
        pass


class GaussianKernel(Kernel):
    def log_kernel(self, u: np.ndarray, center: float) -> np.ndarray:
        return -np.power(u - center, 2.0) / (2.0 * (self.scale ** 2.0))

    def _tune_scale(self, y: float, u: float, p: float):
        self.scale = np.abs(u - y) / norm.ppf(q=((p + 1) / 2))

    def sample(self, size: Optional[int] = None, random_state=None) -> Union[float, np.ndarray]:
        return norm.rvs(loc=0.0, scale=self.scale, size=size, random_state=random_state)


class CauchyKernel(Kernel):
    def log_kernel(self, u: np.ndarray, center: float) -> np.ndarray:
        return -np.log1p(np.power(u - center, 2.0) / (self.scale ** 2.0))

    def _tune_scale(self, y: float, u: float, p: float):
        self.scale = np.abs(u - y) / cauchy.ppf(q=((p + 1) / 2))

    def sample(self, size: Optional[int] = None, random_state=None) -> Union[float, np.ndarray]:
        return cauchy.rvs(loc=0.0, scale=self.scale, size=size, random_state=random_state)


class ProposalDistribution:
    def __init__(self,
                 distribution_f,
                 param_update: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 **kwargs):
        self.distribution_f = distribution_f
        self.param_update = param_update
        self.kwargs = kwargs

    def sample(self, size: Optional[int] = None, random_state=None):
        if self.param_update is not None:
            params = self.param_update(self.kwargs.copy())
        else:
            params = self.kwargs

        return self.distribution_f.rvs(loc=0.0, size=size, random_state=random_state, **params)

    def log_prob(self, x: float, center: float):
        if self.param_update is not None:
            params = self.param_update(self.kwargs.copy())
        else:
            params = self.kwargs

        return self.distribution_f.logpdf(x=x, loc=center, **params)


class MH(abc.ABC):
    def __init__(self,
                 n_samples: int,
                 prior: Dict[str, rv_continuous],
                 proposal: Dict[str, ProposalDistribution],
                 tune: bool = True,
                 tune_interval: int = 100,
                 theta_init: Optional[Dict[str, float]] = None,
                 random_state=None):
        self.n_samples = n_samples
        self.prior = prior
        self.proposal = proposal
        self.tune = tune
        self.tune_interval = tune_interval
        self.theta_init = theta_init
        self.random_state = check_random_state(random_state)

        self._scaling = 1.0
        self._steps_until_tune = tune_interval
        self._accepted = 0

    def do_inference(self, y: np.ndarray) -> List[Dict[str, float]]:
        theta_init = self._mh_init()
        loglik_hat = self._log_likelihood_estimate(y, theta_init)

        thetas = [theta_init]
        loglik_hats = [loglik_hat]
        total_accepted = 0

        for i in range(self.n_samples):
            theta, loglik_hat, accepted = self._mh_step(y, thetas[i], loglik_hats[i])
            thetas.append(theta)
            loglik_hats.append(loglik_hat)
            total_accepted += int(accepted)

            print('Done sample {} of {} ({:.02f}%).'.format(i + 1, self.n_samples, (i + 1) / self.n_samples * 100.0))
            print('Accepted: {} of {} ({:.02f}%) samples so far.'.format(total_accepted, i + 1,
                                                                         total_accepted / (i + 1) * 100.0))

        print('Accepted {} out of {} samples ({:.02f}%).'.format(total_accepted, self.n_samples,
                                                                 total_accepted / self.n_samples * 100.0))

        return thetas

    def _mh_init(self) -> Dict[str, float]:
        if self.theta_init is None:
            return {var_name: dist.rvs(random_state=self.random_state) for var_name, dist in self.prior.items()}
        else:
            return self.theta_init.copy()

    def _mh_step(self, y: np.ndarray, theta_old: Dict[str, float], loglik_hat_old: float) -> Tuple[
        Dict[str, float], float, bool]:
        if not self._steps_until_tune and self.tune:
            self._tune_scale()
            self._steps_until_tune = self.tune_interval
            self._accepted = 0

        delta = self._sample_from_proposal()
        theta_new = {var_name: var_value + delta[var_name] * self._scaling for var_name, var_value in theta_old.items()}

        log_ratio = 0.0

        for var_name, prior in self.prior.items():
            log_ratio += prior.logpdf(theta_new[var_name])
            log_ratio -= prior.logpdf(theta_old[var_name])

        if not np.isfinite(log_ratio):
            return theta_old, loglik_hat_old, False  # Rejected.

        for var_name, proposal in self.proposal.items():
            log_ratio += proposal.log_prob(theta_old[var_name], theta_new[var_name])
            log_ratio -= proposal.log_prob(theta_new[var_name], theta_old[var_name])

        # FIXME: THIS SHOULD APPARENTLY NOT BE HERE.
        # if not np.isfinite(log_ratio):
        #     return theta_old, loglik_hat_old, False  # Rejected.

        loglik_hat_new = self._log_likelihood_estimate(y, theta_new)

        log_ratio += loglik_hat_new
        log_ratio -= loglik_hat_old

        self._steps_until_tune -= 1

        if np.isfinite(log_ratio) and np.log(self.random_state.uniform()) < log_ratio:
            self._accepted += 1
            return theta_new, loglik_hat_new, True  # Accepted.
        else:
            return theta_old, loglik_hat_old, False  # Rejected.

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
        acceptance_rate = self._accepted / self.tune_interval

        if acceptance_rate < 0.001:
            # Reduce by 90 percent.
            self._scaling *= 0.1
        elif acceptance_rate < 0.05:
            # Reduce by 50 percent.
            self._scaling *= 0.5
        elif acceptance_rate < 0.2:
            # Reduce by 10 percent.
            self._scaling *= 0.9
        elif acceptance_rate > 0.95:
            # Increase by a factor of 10.
            self._scaling *= 10.0
        elif acceptance_rate > 0.75:
            # Increase by a factor of 2.
            self._scaling *= 2.0
        elif acceptance_rate > 0.5:
            # Increase by 10 percent.
            self._scaling *= 1.1

    def _sample_from_proposal(self) -> Dict[str, float]:
        return {var_name: dist.sample(random_state=self.random_state) for var_name, dist in
                self.proposal.items()}

    @abc.abstractmethod
    def _log_likelihood_estimate(self, y: np.ndarray, theta: Dict[str, float]) -> float:
        pass


class PMH(MH, abc.ABC):
    """
    Estimates the parameters of the state space model given by
      p(x_{0:T}, y_{1:T}; \theta) = \pi(x_0) \prod_{t=1}^T f(x_t \mid x_{t-1}; \theta) g(y_t \mid x_t; \theta),
    where pi(.) is the prior distribution of the initial state,
          f(. \mid .; \theta) is the transition distribution,
          g(. \mid .; \theta) is the emission distribution.
    The intractable likelihood p(y_{1:T}; \theta) is estimated using the particle filter.

    Assumes the following model:
    x_0 -> x_1 -> x_2 -> ... -> x_T
            |      |             |
           y_1    y_2           y_T
    """

    def __init__(self,
                 n_samples: int,
                 n_particles: int,
                 state_init: np.ndarray,
                 const: Dict[str, float],
                 prior: Dict[str, rv_continuous],
                 proposal: Dict[str, ProposalDistribution],
                 tune: bool = True,
                 tune_interval: int = 100,
                 theta_init: Optional[Dict[str, float]] = None,
                 random_state=None):
        super(PMH, self).__init__(n_samples=n_samples, prior=prior, proposal=proposal, tune=tune,
                                  tune_interval=tune_interval, theta_init=theta_init, random_state=random_state)

        self.n_particles = n_particles
        self.state_init = state_init
        self.const = const

    def _log_likelihood_estimate(self, y: np.ndarray, theta: Dict[str, float]) -> float:
        T = y.shape[0]
        x = np.tile(self.state_init[:, np.newaxis], [1, self.n_particles])
        assert len(x.shape) == 2 and x.shape[1] == self.n_particles

        log_w = np.empty(shape=(T + 1, self.n_particles), dtype=float)
        log_w[0] = -np.log(self.n_particles)

        for t in range(1, T + 1):
            w = np.exp(log_w[t - 1])
            w /= np.sum(w)

            indices = self.random_state.choice(self.n_particles, size=self.n_particles, replace=True, p=w)
            x = self._transition(state=x[:, indices], t=t, theta=theta)

            log_w[t] = self._observation_log_prob(y=y[t - 1], state=x, theta=theta)

        return np.sum(logsumexp(log_w[1:], axis=1)) - T * np.log(self.n_particles)

    @abc.abstractmethod
    def _transition(self, state: np.ndarray, t: int, theta: Dict[str, float]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _observation_log_prob(self, y: float, state: np.ndarray, theta: Dict[str, float]) -> float:
        pass


# TODO: Thinning & burn-in.
# TODO: Increase the trace plot dynamic (non-dynamic proposal scale tuning).
# TODO: Gamma prior, truncated normal proposal.
# TODO: Add an optinal `is_symmetric` parameter to the proposal & filter in the proposal log-prob loop.


# TODO: Store the sampler state so that we can load it and continue sampling some more.
class ABCMH(MH, abc.ABC):
    def __init__(self,
                 n_samples: int,
                 n_particles: int,
                 alpha: float,
                 hpr_p: float,
                 state_init: np.ndarray,
                 const: Dict[str, float],
                 prior: Dict[str, rv_continuous],
                 proposal: Dict[str, ProposalDistribution],
                 kernel: str,
                 noisy_abc: bool = False,
                 tune: bool = True,
                 tune_interval: int = 100,
                 theta_init: Optional[Dict[str, float]] = None,
                 random_state=None):
        assert int(alpha * n_particles) <= n_particles, \
            'the number of covered pseudo-measurements must be at most the number of particles'

        super(ABCMH, self).__init__(n_samples=n_samples, prior=prior, proposal=proposal, tune=tune,
                                    tune_interval=tune_interval, theta_init=theta_init, random_state=random_state)

        self.n_particles = n_particles
        self.n_particles_covered = int(alpha * n_particles)
        self.hpr_p = hpr_p
        self.state_init = state_init
        self.const = const

        if kernel == 'gaussian':
            self.kernel = GaussianKernel()
        elif kernel == 'cauchy':
            self.kernel = CauchyKernel()
        else:
            raise ValueError('Unknown kernel: {}.'.format(kernel))

        self.noisy_abc = noisy_abc

    def _log_likelihood_estimate(self, y: np.ndarray, theta: Dict[str, float]) -> float:
        T = y.shape[0]
        x = np.tile(self.state_init[:, np.newaxis], [1, self.n_particles])
        assert len(x.shape) == 2 and x.shape[1] == self.n_particles

        if self.noisy_abc:
            y = y.copy()

        log_w = np.empty(shape=(T + 1, self.n_particles), dtype=float)
        log_w[0] = -np.log(self.n_particles)

        for t in range(1, T + 1):
            w = np.exp(log_w[t - 1])
            w /= np.sum(w)

            indices = self.random_state.choice(self.n_particles, size=self.n_particles, replace=True, p=w)
            x = self._transition(state=x[:, indices], t=t, theta=theta)
            u = self._measurement_model(state=x, theta=theta)

            u_alpha = self._alphath_closest(u=u, y=y[t - 1])
            self.kernel.tune_scale(y=y[t - 1], u=u_alpha, p=self.hpr_p)

            if self.noisy_abc:
                y[t - 1] += self.kernel.sample(random_state=self.random_state)

            log_w[t] = self.kernel.log_kernel(u=u, center=y[t - 1])

        out = np.sum(logsumexp(log_w[1:], axis=1)) - T * np.log(self.n_particles)

        # Underflow check.
        if out < -20:
            return -1500
        else:
            return out

    def _alphath_closest(self, u: np.ndarray, y: float) -> float:
        # FIXME: This assumes 1D y and u.
        distances_squared = np.power(y - u, 2)

        # Alpha denotes the number of pseudo-measurements covered by the p-HPR of the kernel. However,
        # indexing is 0-based, so we subtract 1 to get the alphath closest pseudo-measurement to y.
        alphath_closest_idx = np.argpartition(distances_squared, kth=self.n_particles_covered - 1, axis=-1)[
            self.n_particles_covered - 1]

        return u[alphath_closest_idx]

    @abc.abstractmethod
    def _transition(self, state: np.ndarray, t: int, theta: Dict[str, float]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _measurement_model(self, state: np.ndarray, theta: Dict[str, float]) -> np.array:
        pass
