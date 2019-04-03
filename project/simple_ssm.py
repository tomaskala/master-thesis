import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
from scipy import stats

from mcmc import ABCMH, PMH, ProposalDistribution
from utils import check_random_state, plot_parameters


class ABCMHSimpleSSM(ABCMH):
    def _transition(self, state: np.ndarray, t: int, theta: Dict[str, float]) -> np.ndarray:
        assert state.shape == (1, self.n_particles)

        X_old = state[0]

        V_n = self.random_state.normal(loc=0.0, scale=np.sqrt(theta['sigma2_v']), size=self.n_particles)
        X_new = X_old / 2 + 25 * (X_old / (1 + np.power(X_old, 2))) + 8 * np.cos(1.2 * t) + V_n

        out = np.array([X_new])
        assert out.shape == state.shape
        return out

    def _measurement_model(self, state: np.ndarray, theta: Dict[str, float]) -> np.array:
        assert state.shape == (1, self.n_particles)

        out = np.power(state, 2) / 20

        assert out.shape == (1, self.n_particles)
        return out


class PMHSimpleSSM(PMH):
    def _transition(self, state: np.ndarray, t: int, theta: Dict[str, float]) -> np.ndarray:
        assert state.shape == (1, self.n_particles)

        X_old = state[0]
        V_n = self.random_state.normal(loc=0.0, scale=np.sqrt(theta['sigma2_v']), size=self.n_particles)
        X_new = X_old / 2 + 25 * (X_old / (1 + np.power(X_old, 2))) + 8 * np.cos(1.2 * t) + V_n

        out = np.array([X_new])
        assert out.shape == state.shape
        return out

    def _observation_log_prob(self, y: np.ndarray, state: np.ndarray, theta: Dict[str, float]) -> float:
        x = state[0]
        loc = np.power(x, 2) / 20
        return stats.norm.logpdf(x=y[0], loc=loc, scale=np.sqrt(self.const['sigma2_w']))


def simulate_xy(path: str, T: int, sigma2_v: float, sigma2_w: float, sigma2_x1: float, random_state=None) -> Tuple[
    np.ndarray, np.ndarray]:
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


def update_truncnorm(center: float, params: Dict[str, Any]) -> Dict[str, Any]:
    params['a'] = (params['a'] - center) / params['scale']
    params['b'] = (params['b'] - center) / params['scale']

    return params


def main():
    # Either 'abcmh' or 'pmh'.
    algorithm = 'abcmh'

    simple_ssm_path = './simple_ssm_{}'.format(algorithm)

    if not os.path.exists(simple_ssm_path):
        os.makedirs(simple_ssm_path)

    const = {
        'sigma2_w': 1.0
    }

    prior = {
        # 'sigma2_v': stats.uniform(0.0, 30.0),
        'sigma2_v': stats.invgamma(a=1.0, scale=10.0),
        # 'sigma2_w': stats.invgamma(a=0.01, scale=0.01)
    }

    # TODO: scale=0.15 works? with tuning.
    proposal = {
        # 'sigma2_v': ProposalDistribution(distribution_f=stats.norm, scale=0.15, is_symmetric=True),
        'sigma2_v': ProposalDistribution(distribution_f=stats.truncnorm, param_update=update_truncnorm, scale=0.15,
                                         a=0.0, b=np.inf)
        # 'sigma2_w': ProposalDistribution(distribution_f=stats.norm, scale=0.08),
    }

    theta_init = {
        'sigma2_v': 20.0,
        # 'sigma2_w': 100.0
    }

    sigma2_v = 10.0
    sigma2_w = 1.0
    sigma2_x1 = 0.5
    sigma_x1 = np.sqrt(sigma2_x1)
    state_init = stats.norm.rvs(loc=0.0, scale=sigma_x1, size=1, random_state=1)

    sampler_path = os.path.join(simple_ssm_path, 'sampler.pickle')

    if os.path.exists(sampler_path):
        with open(sampler_path, mode='rb') as f:
            mcmc = pickle.load(f)
    else:
        if algorithm == 'abcmh':
            mcmc = ABCMHSimpleSSM(n_samples=2000,
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
            mcmc = PMHSimpleSSM(n_samples=2000,  # Paper: 50000 samples, 10000 burn-in.
                                n_particles=500,  # Paper: 5000 particles.
                                state_init=state_init,
                                const=const,
                                prior=prior,
                                proposal=proposal,
                                theta_init=theta_init,
                                random_state=1)

    x, y = simulate_xy(os.path.join(simple_ssm_path, 'simulated_data.pickle'), T=500, sigma2_v=sigma2_v,
                       sigma2_w=sigma2_w, sigma2_x1=sigma2_x1, random_state=1)

    sampled_theta_path = os.path.join(simple_ssm_path, 'sampled_theta.pickle')

    if os.path.exists(sampled_theta_path):
        with open(sampled_theta_path, mode='rb') as f:
            theta = pickle.load(f)
    else:
        theta = mcmc.do_inference(y)

        with open(sampler_path, mode='wb') as f:
            pickle.dump(mcmc, f)

        with open(sampled_theta_path, mode='wb') as f:
            pickle.dump(theta, f)

    pretty_names = {
        'sigma2_v': r'$\sigma^2_v$',
        'sigma2_w': r'$\sigma^2_w$',
    }

    true_values = {
        'sigma2_v': sigma2_v,
        'sigma2_w': sigma2_w
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

    plot_parameters(thetas=theta, pretty_names=pretty_names, true_values=true_values, priors=prior,
                    kernel_scales=mcmc.kernel[0].scale_log if isinstance(mcmc, ABCMH) else None, bins=200, max_lags=100)


if __name__ == '__main__':
    main()
