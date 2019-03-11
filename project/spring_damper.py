import pickle
from typing import Any, Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mcmc import PMH, ProposalDistribution


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


class SpringDamperSimple(PMH):
    def _transition(self, state: np.ndarray, theta: Dict[str, float]) -> np.ndarray:
        v = self.random_state.normal(loc=0.0, scale=1e-2, size=self.n_particles)

        state_old = state.copy()
        s_old = state_old[0]
        sdot_old = state_old[1]

        f_c = self.const['f_c']
        c_0 = theta['c_0']
        k = self.const['k']
        p = self.const['p']

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


def update_truncnorm(params: Dict[str, Any]) -> Dict[str, Any]:
    params['a'] = params['a'] / params['scale']
    params['b'] = params['b'] / params['scale']

    return params


def main_simple():
    np.seterr(all='raise')

    # theta_true = {
    #     'f_c': 0.01,
    #     'c_0': 0.71,
    #     'k': 2.16,
    #     'p': 0.58
    # }

    const = {
        'T_s': 0.1,
        'm': 2,
        'f_c': 0.01,
        'k': 2.16,
        'p': 0.58
    }

    prior = {
        'c_0': stats.gamma(a=2, scale=1)
    }

    proposal = {
        # 'f_c': ProposalDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
        #                             scale=1e-3, a=0.0, b=np.inf),
        'c_0': ProposalDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
                                    scale=1e-2, a=0.0, b=1.0),
        # 'k': ProposalDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
        #                           scale=1e-2, a=0.0, b=np.inf),
        # 'p': ProposalDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
        #                           scale=1e-2, a=0.0, b=1.0),
    }

    # s, sdot
    state_init = np.array([0.5, 0.0])

    with h5py.File('./obs.nc', mode='r') as obs_file:
        y = obs_file['y'][()].reshape(-1)

    pmh = SpringDamperSimple(n_samples=100,
                             n_particles=256,
                             state_init=state_init,
                             const=const,
                             prior=prior,
                             proposal=proposal,
                             random_state=1)

    theta = pmh.do_inference(y)

    with open('./sampled_theta.pickle', mode='wb') as f:
        pickle.dump(theta, f)


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
        'f_c': ProposalDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
                                    scale=1e-3, a=0.0, b=np.inf),
        'c_0': ProposalDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
                                    scale=1e-2, a=0.0, b=1.0),
        'k': ProposalDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
                                  scale=1e-2, a=0.0, b=np.inf),
        'p': ProposalDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm,
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
        if len(sampled[name]) == 0:
            continue

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

    plt.hist(log_likelihood, bins=20, density=True)
    plt.axvline(np.mean(log_likelihood), color='red', lw=2)
    plt.show()


if __name__ == '__main__':
    # main_simple()
    main()
    plot_parameters()
    # plot_likelihood()
