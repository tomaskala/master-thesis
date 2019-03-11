import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rv_continuous

from utils import check_random_state


class Kernel(abc.ABC):
    pass


class ProposalDistribution:
    def __init__(self,
                 distribution_f,
                 param_update: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda x: x,
                 **kwargs):
        self.distribution_f = distribution_f
        self.param_update = param_update
        self.kwargs = kwargs

    def sample(self, size: Optional[int] = None, random_state=None):
        params = self.param_update(self.kwargs.copy())
        return self.distribution_f.rvs(loc=0.0, size=size, random_state=random_state, **params)

    def log_prob(self, x: float, center: float):
        params = self.param_update(self.kwargs.copy())
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
            self.tune_scale()
            self._steps_until_tune = self.tune_interval
            self._accepted = 0

        delta = self._sample_from_proposal()
        theta_new = {var_name: var_value + delta[var_name] * self._scaling for var_name, var_value in theta_old.items()}
        loglik_hat_new = self._log_likelihood_estimate(y, theta_new)

        log_ratio = loglik_hat_new - loglik_hat_old

        for var_name, prior in self.prior.items():
            log_ratio += prior.logpdf(theta_new[var_name])
            log_ratio -= prior.logpdf(theta_old[var_name])

        for var_name, proposal in self.proposal.items():
            log_ratio += proposal.log_prob(theta_old[var_name], theta_new[var_name])
            log_ratio -= proposal.log_prob(theta_new[var_name], theta_old[var_name])

        self._steps_until_tune -= 1

        if np.isfinite(log_ratio) and np.log(self.random_state.uniform()) < log_ratio:
            self._accepted += 1
            return theta_new, loglik_hat_new, True  # Accepted.
        else:
            return theta_old, loglik_hat_old, False  # Rejected.

    def tune_scale(self):
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

        # Switch statement
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
    def __init__(self,
                 n_samples: int,
                 n_particles: int,
                 state_init: Union[Callable[[int], np.ndarray], np.ndarray],
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

        if callable(self.state_init):
            x = self.state_init(self.n_particles)
        else:
            x = np.tile(self.state_init[:, np.newaxis], [1, self.n_particles])

        assert len(x.shape) == 2 and x.shape[1] == self.n_particles

        log_w = np.empty(shape=(T, self.n_particles), dtype=float)
        log_w[0] = self._observation_log_prob(y=y[0], state=x, theta=theta)

        for t in range(1, T):
            w = np.exp(log_w[t - 1])
            w /= np.sum(w)

            indices = self.random_state.choice(self.n_particles, size=self.n_particles, replace=True, p=w)

            x = self._transition(state=x[:, indices], n=t + 1, theta=theta)
            log_w[t] = self._observation_log_prob(y=y[t], state=x, theta=theta)

        return np.sum(logsumexp(log_w, axis=1)) - T * np.log(self.n_particles)

    @abc.abstractmethod
    def _transition(self, state: np.ndarray, n: int, theta: Dict[str, float]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _observation_log_prob(self, y: float, state: np.ndarray, theta: Dict[str, float]) -> float:
        pass


class ABCMH(MH, abc.ABC):
    def __init__(self,
                 n_samples: int,
                 n_particles: int,
                 state_init: Union[Callable[[int], np.ndarray], np.ndarray],
                 const: Dict[str, float],
                 prior: Dict[str, rv_continuous],
                 proposal: Dict[str, ProposalDistribution],
                 kernel: Kernel,
                 tune: bool = True,
                 tune_interval: int = 100,
                 theta_init: Optional[Dict[str, float]] = None,
                 random_state=None):
        super(ABCMH, self).__init__(n_samples=n_samples, prior=prior, proposal=proposal, tune=tune,
                                    tune_interval=tune_interval, theta_init=theta_init, random_state=random_state)

        self.n_particles = n_particles
        self.state_init = state_init
        self.const = const
        self.kernel = kernel

    def _log_likelihood_estimate(self, y: np.ndarray, theta: Dict[str, float]) -> float:
        T = y.shape[0]

        if callable(self.state_init):
            x = self.state_init(self.n_particles)
        else:
            x = np.tile(self.state_init[:, np.newaxis], [1, self.n_particles])

        assert len(x.shape) == 2 and x.shape[1] == self.n_particles

        log_w = np.empty(shape=(T, self.n_particles), dtype=float)
        # log_w[0] = self._observation_log_prob(y=y[0], state=x, theta=theta)  # TODO: Set correctly.

        for t in range(1, T):
            w = np.exp(log_w[t - 1])
            w /= np.sum(w)

            indices = self.random_state.choice(self.n_particles, size=self.n_particles, replace=True, p=w)

            x = self._transition(state=x[:, indices], n=t + 1, theta=theta)
            # log_w[t] = self._observation_log_prob(y=y[t], state=x, theta=theta)  # TODO: Set correctly.

        return np.sum(logsumexp(log_w, axis=1)) - T * np.log(self.n_particles)

    @abc.abstractmethod
    def _transition(self, state: np.ndarray, n: int, theta: Dict[str, float]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _measurement_model(self, state: np.ndarray, theta: Dict[str, float]) -> float:
        pass
