import os
import pickle
from typing import Dict, Tuple

import numpy as np
from scipy import stats

from pmcmc import PMH, ProposalDistribution
from utils import check_random_state, plot_parameters


class SimpleSSM(PMH):
    def _transition(self, state: np.ndarray, n: int, theta: Dict[str, float]) -> np.ndarray:
        assert state.shape == (1, self.n_particles)

        X_old = state[0]
        V_n = self.random_state.normal(loc=0.0, scale=np.sqrt(theta['sigma2_v']), size=self.n_particles)
        X_new = X_old / 2 + 25 * (X_old / (1 + np.power(X_old, 2))) + 8 * np.cos(1.2 * n) + V_n

        out = np.array([X_new])
        assert out.shape == state.shape
        return out

    def _observation_log_prob(self, y: float, state: np.ndarray, theta: Dict[str, float]) -> float:
        x = state[0]
        loc = np.power(x, 2) / 20
        return stats.norm.logpdf(x=y, loc=loc, scale=np.sqrt(theta['sigma2_w']))


def simulate_xy(path: str, T: int, sigma2_v: float, sigma2_w: float, sigma2_x1: float, random_state=None) -> Tuple[
    np.ndarray, np.ndarray]:
    if os.path.exists(path):
        with open(path, mode='rb') as f:
            return pickle.load(f)
    else:
        random_state = check_random_state(random_state)
        x = np.empty(shape=T, dtype=float)
        y = np.empty(shape=T, dtype=float)
        x[0] = random_state.normal(loc=0.0, scale=np.sqrt(sigma2_x1))
        w = random_state.normal(loc=0.0, scale=np.sqrt(sigma2_w))
        y[0] = np.power(x[0], 2) / 20 + w

        for n in range(1, T):
            v = random_state.normal(loc=0.0, scale=np.sqrt(sigma2_v))
            x[n] = x[n - 1] / 2 + 25 * (x[n - 1] / (1 + np.power(x[n - 1], 2))) + 8 * np.cos(1.2 * (n + 1)) + v

            w = random_state.normal(loc=0.0, scale=np.sqrt(sigma2_w))
            y[n] = np.power(x[n], 2) / 20 + w

        with open(path, mode='wb') as f:
            pickle.dump((x, y), f)

        return x, y


def main():
    simple_ssm_path = './simple_ssm'

    if not os.path.exists(simple_ssm_path):
        os.makedirs(simple_ssm_path)

    const = {}

    prior = {
        'sigma2_v': stats.invgamma(0.01, scale=0.01),
        'sigma2_w': stats.invgamma(0.01, scale=0.01)
    }

    proposal = {
        'sigma2_v': ProposalDistribution(distribution_f=stats.norm, scale=0.15),
        'sigma2_w': ProposalDistribution(distribution_f=stats.norm, scale=0.08),
    }

    theta_init = {
        'sigma2_v': 100.0,
        'sigma2_w': 100.0
    }

    sigma2_v = 10.0
    sigma2_w = 1.0
    sigma2_x1 = 0.5
    sigma_x1 = np.sqrt(sigma2_x1)

    def state_init(n_particles):
        return stats.norm.rvs(loc=0.0, scale=sigma_x1, size=(1, n_particles), random_state=1)

    pmh = SimpleSSM(n_samples=10000,  # Paper: 50000 samples, 10000 burn-in.
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
        theta = pmh.do_inference(y)

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

    plot_parameters(theta, pretty_names=pretty_names, true_values=true_values, priors=prior)


if __name__ == '__main__':
    main()
