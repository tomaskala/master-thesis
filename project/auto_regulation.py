"""
Bayesian Parameter Inference for Stochastic Biochemical Network Models Using Particle Markov Chain Monte Carlo,
source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3262293/pdf/rsfs20110047.pdf
"""

import os
import pickle
from typing import Dict, Tuple

import numpy as np
from scipy import stats

from mcmc import ABCMH, PMH, ProposalDistribution
from utils import check_random_state, plot_parameters


def batch_diag(array: np.ndarray) -> np.ndarray:
    """
    Apply the `diag` operation on each row of the 2D input array, producing a 3D array with the individual
    diagonal matrices stored along the first dimension.
    :param array: array shaped (M, N)
    :return: array shaped (N, M, M)
    """
    rows, cols = array.shape
    out = np.zeros((cols, rows, rows), dtype=array.dtype)
    diag = np.arange(rows)
    out[:, diag, diag] = array.T

    return out


class ABCMHAutoRegulation(ABCMH):
    def _transition(self, state: np.ndarray, t: int, theta: Dict[str, float]) -> np.ndarray:
        S = self.const['S']
        m = self.const['m']
        observation_std = self.const['observation_std']
        dt = 1.0 / m

        for i in range(m):
            h = self._hazard_function(state, theta)
            assert h.shape == (8, self.n_particles)

            h_diag = batch_diag(h.T)
            assert h_diag.shape == (self.n_particles, 8, 8)

            alpha = S @ h * dt
            assert alpha.shape == (4, self.n_particles)

            beta = S @ h_diag @ S.T * dt
            assert beta.shape == (self.n_particles, 4, 4)

            w = stats.norm.rvs(loc=0.0, scale=observation_std, size=state.T.shape, random_state=self.random_state)
            assert w.shape == (4, self.n_particles)

            state = alpha + (np.sqrt(beta) @ w[..., np.newaxis]).reshape(self.n_particles, 4).T
            assert state.shape == (4, self.n_particles)

        return state

    def _measurement_model(self, state: np.ndarray, theta: Dict[str, float]) -> np.array:
        return state[1] + 2 * state[2]

    def _hazard_function(self, state: np.ndarray, theta: Dict[str, float]):
        c1 = np.exp(theta['lc1'])
        c2 = np.exp(theta['lc2'])
        c3 = np.exp(theta['lc3'])
        c4 = np.exp(theta['lc4'])
        c5 = self.const['c5']
        c6 = self.const['c6']
        c7 = np.exp(theta['lc7'])
        c8 = np.exp(theta['lc8'])

        rna = state[0]
        p = state[1]
        p2 = state[2]
        dna = state[3]

        out = np.array([
            c1 * dna * p2,
            c2 * (self.const['k'] - dna),
            c3 * dna,
            c4 * rna,
            c5 * p * (p - 1) / 2,
            c6 * p2,
            c7 * rna,
            c8 * p
        ])

        assert out.shape == (8, self.n_particles)
        return out


class PMHAutoRegulation(PMH):
    def _transition(self, state: np.ndarray, t: int, theta: Dict[str, float]) -> np.ndarray:
        S = self.const['S']
        m = self.const['m']
        observation_std = self.const['observation_std']
        dt = 1.0 / m

        for i in range(m):
            h = self._hazard_function(state, theta)
            assert h.shape == (8, self.n_particles)

            h_diag = batch_diag(h.T)
            assert h_diag.shape == (self.n_particles, 8, 8)

            alpha = S @ h * dt
            assert alpha.shape == (4, self.n_particles)

            beta = S @ h_diag @ S.T * dt
            assert beta.shape == (self.n_particles, 4, 4)

            w = stats.norm.rvs(loc=0.0, scale=observation_std, size=state.T.shape, random_state=self.random_state)
            assert w.shape == (4, self.n_particles)

            state = alpha + (np.sqrt(beta) @ w[..., np.newaxis]).reshape(self.n_particles, 4).T
            assert state.shape == (4, self.n_particles)

        return state

    def _observation_log_prob(self, y: np.ndarray, state: np.ndarray, theta: Dict[str, float]) -> float:
        loc = state[1] + 2 * state[2]
        return stats.norm.logpdf(y=y[0], loc=loc, scale=self.const['observation_std'])

    def _hazard_function(self, state: np.ndarray, theta: Dict[str, float]):
        c1 = np.exp(theta['lc1'])
        c2 = np.exp(theta['lc2'])
        c3 = np.exp(theta['lc3'])
        c4 = np.exp(theta['lc4'])
        c5 = self.const['c5']
        c6 = self.const['c6']
        c7 = np.exp(theta['lc7'])
        c8 = np.exp(theta['lc8'])

        rna = state[0]
        p = state[1]
        p2 = state[2]
        dna = state[3]

        out = np.array([
            c1 * dna * p2,
            c2 * (self.const['k'] - dna),
            c3 * dna,
            c4 * rna,
            c5 * p * (p - 1) / 2,
            c6 * p2,
            c7 * rna,
            c8 * p
        ])

        assert out.shape == (8, self.n_particles)
        return out


# TODO: Gillespie algorithm.
def simulate_xy(path: str, T: int, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(path):
        with open(path, mode='rb') as f:
            return pickle.load(f)
    else:
        random_state = check_random_state(random_state)
        x = np.empty(shape=T, dtype=float)
        y = np.empty(shape=T, dtype=float)
        x_0 = random_state.normal(loc=0.0, scale=np.sqrt(sigma2_x1))

        x_prev = x_0

        for n in range(T):
            v = random_state.normal(loc=0.0, scale=np.sqrt(sigma2_v))
            x[n] = x_prev / 2 + 25 * (x_prev / (1 + np.power(x_prev, 2))) + 8 * np.cos(1.2 * (n + 1)) + v
            x_prev = x[n]

            w = random_state.normal(loc=0.0, scale=np.sqrt(sigma2_w))
            y[n] = np.power(x[n], 2) / 20 + w

        with open(path, mode='wb') as f:
            pickle.dump((x, y[np.newaxis, :]), f)

        return np.append(x_0, x), y[np.newaxis, :]


def main():
    # Either 'abcmh' or 'pmh'.
    algorithm = 'abcmh'

    auto_regulation_path = './auto_regulation_{}'.format(algorithm)

    if not os.path.exists(auto_regulation_path):
        os.makedirs(auto_regulation_path)

    const = {
        'k': 10,
        'm': 5,
        'c5': 0.1,
        'c6': 0.9,
        'observation_std': 2.0,
        'S': np.array([
            [0, 0, 1, 0, 0, 0, -1, 0],
            [0, 0, 0, 1, -2, 2, 0, -1],
            [-1, 1, 0, 0, 1, -1, 0, 0],
            [-1, 1, 0, 0, 0, 0, 0, 0]
        ])
    }

    # log(c_i) ~ U(-7,2), but in SciPy, the uniform distribution is parameterized as U(loc,loc+scale).
    prior = {
        'lc1': stats.uniform(loc=-7.0, scale=9.0),
        'lc2': stats.uniform(loc=-7.0, scale=9.0),
        'lc3': stats.uniform(loc=-7.0, scale=9.0),
        'lc4': stats.uniform(loc=-7.0, scale=9.0),
        'lc7': stats.uniform(loc=-7.0, scale=9.0),
        'lc8': stats.uniform(loc=-7.0, scale=9.0)
    }

    proposal = {
        'lc1': ProposalDistribution(distribution_f=stats.norm, scale=1.0),
        'lc2': ProposalDistribution(distribution_f=stats.norm, scale=1.0),
        'lc3': ProposalDistribution(distribution_f=stats.norm, scale=1.0),
        'lc4': ProposalDistribution(distribution_f=stats.norm, scale=1.0),
        'lc7': ProposalDistribution(distribution_f=stats.norm, scale=1.0),
        'lc8': ProposalDistribution(distribution_f=stats.norm, scale=1.0)

    }

    theta_init = None
    state_init = np.array([8, 8, 8, 5])

    sampler_path = os.path.join(auto_regulation_path, 'sampler.pickle')

    if os.path.exists(sampler_path):
        with open(sampler_path, mode='rb') as f:
            mcmc = pickle.load(f)
    else:
        if algorithm == 'abcmh':
            mcmc = ABCMHAutoRegulation(n_samples=2000,
                                       n_particles=500,
                                       alpha=0.9,
                                       hpr_p=0.95,
                                       state_init=state_init,
                                       const=const,
                                       prior=prior,
                                       proposal=proposal,
                                       kernel='gaussian',
                                       noisy_abc=False,
                                       theta_init=theta_init,
                                       random_state=1,
                                       tune=True)
        else:
            mcmc = PMHAutoRegulation(n_samples=2000,
                                     n_particles=500,
                                     state_init=state_init,
                                     const=const,
                                     prior=prior,
                                     proposal=proposal,
                                     theta_init=theta_init,
                                     random_state=1)

    x, y = simulate_xy(os.path.join(auto_regulation_path, 'simulated_data.pickle'), T=100, random_state=1)
    sampled_theta_path = os.path.join(auto_regulation_path, 'sampled_theta.pickle')

    if os.path.exists(sampled_theta_path):
        with open(sampled_theta_path, mode='rb') as f:
            theta = pickle.load(f)
    else:
        theta = mcmc.do_inference(y)

        with open(sampler_path, mode='wb') as f:
            pickle.dump(mcmc, f)

        with open(sampled_theta_path, mode='wb') as f:
            pickle.dump(theta, f)

    transforms = {
        'lc1': np.exp,
        'lc2': np.exp,
        'lc3': np.exp,
        'lc4': np.exp,
        'lc7': np.exp,
        'lc8': np.exp
    }

    pretty_names = {
        'lc1': r'$c_1$',
        'lc2': r'$c_1$',
        'lc3': r'$c_1$',
        'lc4': r'$c_1$',
        'lc7': r'$c_1$',
        'lc8': r'$c_1$'
    }

    true_values = {
        'lc1': np.log(0.1),
        'lc2': np.log(0.7),
        'lc3': np.log(0.35),
        'lc4': np.log(0.2),
        'lc7': np.log(0.3),
        'lc8': np.log(0.1)
    }

    # # Histogram detail.
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.title('Histogram')
    #
    # burn_in = 0
    # step = 1
    # bins = 100
    # plt.ylim((0.0, 0.00175))
    #
    # params2samples = {param_name: [] for param_name in theta[0].keys()}
    #
    # for thetaa in theta:
    #     for param_name, param_value in thetaa.items():
    #         params2samples[param_name].append(param_value)
    #
    # for param_name, param_values in params2samples.items():
    #
    #     plt.hist(param_values[burn_in::step], density=True, bins=bins)
    #
    #     x = np.linspace(np.min(param_values[burn_in::step]), np.max(param_values[burn_in::step]), 100)
    #     plt.plot(x, prior[param_name].pdf(x), color='green')
    #
    #     if pretty_names is not None and param_name in pretty_names:
    #         pretty_name = pretty_names[param_name]
    #     else:
    #         pretty_name = param_name
    #
    #     title = '{}, mean: {:.03f}'.format(pretty_name, np.mean(param_values[burn_in::step]))
    #
    #     if true_values is not None and param_name in true_values:
    #         plt.axvline(true_values[param_name], color='red', lw=2)
    #         title += ', true value: {:.03f}'.format(true_values[param_name])
    #
    #         plt.suptitle(title)
    #
    #     plt.show()

    plot_parameters(thetas=theta, transforms=transforms, pretty_names=pretty_names, true_values=true_values,
                    priors=prior, kernel_scales=mcmc.kernel[0].scale_log if isinstance(mcmc, ABCMH) else None, bins=200,
                    max_lags=100)


if __name__ == '__main__':
    main()
