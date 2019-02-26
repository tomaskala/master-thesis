import abc
import numbers
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from scipy.stats._distn_infrastructure import rv_continuous


def _check_random_state(random_state) -> np.random.RandomState:
    if random_state is None or random_state is np.random:
        return np.random.mtrand._rand
    if isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState instance.'.format(random_state))


class PMH(abc.ABC):
    def __init__(self,
                 n_samples: int,  # M in the paper.
                 n_particles: int,  # N in the paper.
                 state_init: np.ndarray,
                 const: Dict[str, float],
                 prior: Dict[str, rv_continuous],
                 proposal: Dict[str, 'CenteredDistribution'],
                 random_state=None):
        self.n_samples = n_samples
        self.n_particles = n_particles
        self.state_init = state_init
        self.const = const
        self.prior = prior
        self.proposal = proposal
        self.random_state = _check_random_state(random_state)

    def do_inference(self, y: np.ndarray) -> List[Dict[str, float]]:
        theta_init = self._mh_init()
        log_z_hat_init = self._bootstrap_particle_filter(y, theta_init)

        thetas = [theta_init]
        log_z_hats = [log_z_hat_init]
        total_accepted = 0

        for i in range(self.n_samples):
            theta, log_z_hat, accepted = self._mh_step(y, thetas[i], log_z_hats[i])
            thetas.append(theta)
            log_z_hats.append(log_z_hat)
            total_accepted += int(accepted)

            print('Done sample {} of {} ({:.02f}%).'.format(i + 1, self.n_samples, (i + 1) / self.n_samples * 100.0))
            print('Accepted: {} of {} ({:.02f}%) samples so far.'.format(total_accepted, i + 1,
                                                                         total_accepted / (i + 1) * 100.0))

        print('Accepted {} out of {} samples ({:.02f}%).'.format(total_accepted, self.n_samples,
                                                                 total_accepted / self.n_samples * 100.0))

        return thetas

    # TODO: Also return x as the smoother.
    def _bootstrap_particle_filter(self, y: np.ndarray, theta: Dict[str, float]) -> float:
        T = y.shape[0]
        x = np.tile(self.state_init[:, np.newaxis], [1, self.n_particles])
        assert x.shape == (self.state_init.shape[0], self.n_particles)

        log_w = np.empty(shape=(T + 1, self.n_particles), dtype=float)
        log_w[0] = -np.log(self.n_particles)

        for t in range(1, T + 1):
            w = np.exp(log_w[t - 1])
            w /= np.sum(w)

            # TODO: Resample only after every 10th step. Or try stratified resampling.
            # Then, try a simple model with particle.
            # Then, try a linear model with Kalman instead of particle.
            indices = self.random_state.choice(self.n_particles, size=self.n_particles, replace=True, p=w)

            x = self._transition(state=x[:, indices], theta=theta)
            log_w[t] = self._observation_log_prob(y=y[t - 1], state=x, theta=theta)

        # Return the log-likelihood estimate. Ignore the initial uniform weights.
        return np.sum(logsumexp(log_w[1:], axis=1)) - T * np.log(self.n_particles)

    def _mh_init(self) -> Dict[str, float]:
        return {var_name: dist.rvs(random_state=self.random_state) for var_name, dist in self.prior.items()}

    def _mh_step(self, y: np.ndarray, theta_old: Dict[str, float], log_z_hat_old: float) -> Tuple[
        Dict[str, float], float, bool]:
        theta_new = self._sample_from_proposal(theta_old)
        log_z_hat_new = self._bootstrap_particle_filter(y, theta_new)

        log_ratio = log_z_hat_new - log_z_hat_old

        for var_name, prior in self.prior.items():
            log_ratio += prior.logpdf(theta_new[var_name])
            log_ratio -= prior.logpdf(theta_old[var_name])

        for var_name, proposal in self.proposal.items():
            log_ratio += proposal.log_prob(theta_old[var_name], theta_new[var_name])
            log_ratio -= proposal.log_prob(theta_new[var_name], theta_old[var_name])

        if np.isfinite(log_ratio) and np.log(self.random_state.uniform()) < log_ratio:
            return theta_new, log_z_hat_new, True  # Accepted.
        else:
            return theta_old, log_z_hat_old, False  # Rejected.

    def _sample_from_proposal(self, theta: Dict[str, float]) -> Dict[str, float]:
        return {var_name: dist.sample(center=theta[var_name], random_state=self.random_state) for var_name, dist in
                self.proposal.items()}

    @abc.abstractmethod
    def _transition(self, state: np.ndarray, theta: Dict[str, float]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _observation_log_prob(self, y: float, state: np.ndarray, theta: Dict[str, float]) -> float:
        pass


class CenteredDistribution:
    def __init__(self, distribution_f,
                 param_update: Callable[[float, Dict[str, Any]], Dict[str, Any]] = lambda x, y: y,
                 **kwargs):
        self.distribution_f = distribution_f
        self.param_update = param_update
        self.kwargs = kwargs

    def sample(self, center: float, size: Optional[int] = None, random_state=None):
        params = self.param_update(center, self.kwargs.copy())
        return self.distribution_f.rvs(loc=center, size=size, random_state=random_state, **params)

    def log_prob(self, x: float, center: float):
        params = self.param_update(center, self.kwargs.copy())
        return self.distribution_f.logpdf(x=x, loc=center, **params)


class SpringDamper(PMH):
    def _transition(self, state: np.ndarray, theta: Dict[str, float]) -> np.ndarray:
        v = self.random_state.normal(loc=0.0, scale=1e-2, size=self.n_particles)

        state_old = state.copy()
        s_old = state_old[0]
        sdot_old = state_old[1]

        f_c = theta['f_c']
        c_0 = theta['c_0']
        k = theta['k']
        p = theta['p']

        T_s = self.const['T_s']
        m = self.const['m']

        s = s_old + T_s * sdot_old
        sdot = sdot_old + (T_s / m) * (
                -f_c * np.sign(sdot_old) - c_0 * sdot_old - k * np.sign(s_old) * np.power(np.abs(s_old), p)) + v

        out = np.array([s, sdot])
        assert out.shape == state.shape
        return out

    def _observation_log_prob(self, y: float, state: np.ndarray, theta: Dict[str, float]) -> float:
        out = stats.norm.logpdf(x=y, loc=state[0], scale=1e-1)
        assert out.shape == (self.n_particles,)
        return out


def update_truncnorm(center: float, params: Dict[str, Any]) -> Dict[str, Any]:
    params['a'] = (params['a'] - center) / params['scale']
    params['b'] = (params['b'] - center) / params['scale']

    return params


def main():
    # np.seterr(all='raise')

    const = {
        'T_s': 0.1,
        'm': 2
    }

    prior = {
        'f_c': stats.gamma(a=2, scale=0.01),
        'c_0': stats.gamma(a=2, scale=1),
        'k': stats.gamma(a=4, scale=0.3),
        'p': stats.uniform(loc=0, scale=1)
    }

    proposal = {
        'f_c': CenteredDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
                                    scale=1e-3, a=0.0, b=np.inf),
        'c_0': CenteredDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
                                    scale=1e-2, a=0.0, b=1.0),
        'k': CenteredDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
                                  scale=1e-2, a=0.0, b=np.inf),
        'p': CenteredDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
                                  scale=1e-2, a=0.0, b=1.0),
    }

    # s, sdot
    state_init = np.array([0.5, 0.0])

    with h5py.File('./obs.nc', mode='r') as obs_file:
        y = obs_file['y'][()].reshape(-1)

    pmh = SpringDamper(n_samples=1000,
                       n_particles=256,
                       state_init=state_init,
                       const=const,
                       prior=prior,
                       proposal=proposal,
                       random_state=1)

    # theta_true = {'f_c': 0.01, 'c_0': 0.71, 'k': 2.16, 'p': 0.58}
    # x1 = pmh._transition(state=np.tile(state_init[:, np.newaxis], [1, 256]), theta=theta_true)
    #
    # x2 = pmh._transition(state=x1, theta=theta_true)
    #
    # y1 = pmh._observation_log_prob(y[0], state=x2, theta=theta_true)
    #
    # print(state_init)
    # print()
    # print(x1)
    # print()
    # print(y1)
    # # print()
    # # print(x2)
    # exit()

    # theta_true = {
    #     'f_c': 0.01,
    #     'c_0': 0.71,
    #     'k': 2.16,
    #     'p': 0.58
    # }
    #
    # samples = []
    #
    # for t in range(1000):
    #     print(t)
    #     z_hat = pmh._bootstrap_particle_filter(y, theta_true)
    #
    #     samples.append(z_hat)
    #
    # with open('./sampled_likelihood.pickle', mode='wb') as f:
    #     pickle.dump(np.array(samples), f)

    theta = pmh.do_inference(y)

    with open('./sampled_theta.pickle', mode='wb') as f:
        pickle.dump(theta, f)


def plot_parameters():
    import matplotlib.pyplot as plt

    with open('./sampled_theta.pickle', mode='rb') as f:
        theta = pickle.load(f)

    sampled = {
        'f_c': [],
        'c_0': [],
        'k': [],
        'p': []
    }

    for params in theta:
        for k, v in params.items():
            sampled[k].append(v)

    sampled['f_c'] = np.array(sampled['f_c'])
    sampled['c_0'] = np.array(sampled['c_0'])
    sampled['k'] = np.array(sampled['k'])
    sampled['p'] = np.array(sampled['p'])

    prior = {
        'f_c': stats.gamma(a=2, scale=0.01),
        'c_0': stats.gamma(a=2, scale=1),
        'k': stats.gamma(a=4, scale=0.3),
        'p': stats.uniform(loc=0, scale=1)
    }

    lower = {
        'f_c': 0,
        'c_0': 0,
        'k': 0,
        'p': 0
    }

    upper = {
        'f_c': 0.1,
        'c_0': 3,
        'k': 4,
        'p': 2
    }

    true_value = {
        'f_c': 0.01,
        'c_0': 0.71,
        'k': 2.16,
        'p': 0.58
    }

    for i, name in enumerate(sampled):
        plt.subplot(2, 2, i + 1)
        plt.hist(sampled[name], density=True)
        plt.title('{}, mean: {:.03f}, true value: {:.03f}'.format(name, np.mean(sampled[name]), true_value[name]))

        x = np.linspace(lower[name], upper[name], 100)
        plt.plot(x, prior[name].pdf(x))

        plt.axvline(x=true_value[name], color='red', lw=2)

    plt.show()


def plot_likelihood():
    with open('./sampled_likelihood.pickle', mode='rb') as f:
        log_likelihood = pickle.load(f)

    import matplotlib.pyplot as plt

    plt.hist(log_likelihood, bins=20, density=True)
    plt.axvline(np.mean(log_likelihood), color='red', lw=2)
    plt.show()


if __name__ == '__main__':
    main()
    plot_parameters()
    # plot_likelihood()
