import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from mcmc import Distribution, MetropolisHastingsABC, MetropolisHastingsPF, Prior, Proposal
from utils import check_random_state


class ABCSimpleSSM(MetropolisHastingsABC):
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        V_n = self.random_state.normal(loc=0.0, scale=np.sqrt(theta[0]), size=(self.n_particles, 1))
        x_new = x / 2 + 25 * (x / (1 + np.power(x, 2))) + 8 * np.cos(1.2 * t) + V_n
        assert x.shape == x_new.shape, '{} != {}'.format(x.shape, x_new.shape)
        return x_new

    def _measurement_model(self, x: np.ndarray, theta: np.ndarray) -> np.array:
        assert x.shape == (self.n_particles, 1)
        return np.power(x[:, 0], 2) / 20


class ParticleSimpleSSM(MetropolisHastingsPF):
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        V_n = self.random_state.normal(loc=0.0, scale=np.sqrt(theta[0]), size=(self.n_particles, 1))
        x_new = x / 2 + 25 * (x / (1 + np.power(x, 2))) + 8 * np.cos(1.2 * t) + V_n
        assert x.shape == x_new.shape, '{} != {}'.format(x.shape, x_new.shape)
        return x_new

    def _observation_log_prob(self, y: np.ndarray, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        log_prob = np.sum(stats.norm.logpdf(y, np.power(x, 2) / 20, np.sqrt(self.const['sigma2_w'])), axis=1)
        assert log_prob.ndim == 1 and log_prob.shape[0] == self.n_particles
        return log_prob


def simulate_xy(path: str, T: int, sigma2_v: float, sigma2_w: float, sigma2_x1: float, random_state=None):
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

        x = np.append(x_0, x)
        # y = y[:, np.newaxis]

        with open(path, mode='wb') as f:
            pickle.dump((x, y), f)

        return x, y


def main():
    algorithm = 'abcmh'
    path = './simple_ssm_{}'.format(algorithm)
    random_state = check_random_state(1)

    if not os.path.exists(path):
        os.makedirs(path)

    n_samples = 2000
    n_particles = 500
    burn_in = 0
    thinning = 10

    T = 500
    sigma2_v = 10.0
    sigma2_w = 1.0
    sigma2_x1 = 0.5
    state_init = stats.norm.rvs(loc=0.0, scale=np.sqrt(sigma2_x1), size=(n_particles, 1), random_state=random_state)
    x, y = simulate_xy(os.path.join(path, 'simulated_data.pickle'), T=T, sigma2_v=sigma2_v,
                       sigma2_w=sigma2_w, sigma2_x1=sigma2_x1, random_state=1)

    const = {
        'sigma2_w': sigma2_w
    }

    prior = Prior([
        stats.invgamma(a=1.0, scale=10.0)
    ])

    proposal = Proposal([
        Distribution(stats.truncnorm, truncnorm=True, scale=0.8, a=0.0, b=np.inf)
    ])

    theta_init = np.array([20.0])
    random_state = check_random_state(1)

    if algorithm == 'abcmh':
        alpha = 0.9
        hpr_p = 0.95
        kernel = 'gaussian'

        mcmc = ABCSimpleSSM(
            n_samples=n_samples,
            n_particles=n_particles,
            alpha=alpha,
            hpr_p=hpr_p,
            state_init=state_init,
            const=const,
            kernel=kernel,
            prior=prior,
            proposal=proposal,
            tune=False,
            theta_init=theta_init,
            random_state=random_state
        )
    else:
        mcmc = ParticleSimpleSSM(
            n_samples=n_samples,
            n_particles=n_particles,
            state_init=state_init,
            const=const,
            prior=prior,
            proposal=proposal,
            tune=False,
            theta_init=theta_init,
            random_state=random_state
        )

    sampled_theta_path = os.path.join(path, 'sampled_theta.pickle')

    if os.path.exists(sampled_theta_path):
        with open(sampled_theta_path, 'rb') as f:
            theta = pickle.load(f)
    else:
        theta = mcmc.do_inference(y)

        with open(sampled_theta_path, 'wb') as f:
            pickle.dump(theta, f)

    theta = theta[burn_in::thinning]
    truth = np.array([sigma2_v])
    pretty_names = [r'$\sigma_v^2$']

    for i in range(theta.shape[1]):
        param_name = pretty_names[i]
        param_values = theta[:, i]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        plt.suptitle(param_name)

        ax1.set_title('Trace plot')
        ax1.plot(param_values)
        ax1.axhline(truth[i], color='red', lw=2)

        plot_acf(param_values, ax=ax2)

        ax3.set_title('Histogram')
        ax3.hist(param_values, density=True, bins=30)
        ax3.axvline(truth[i], color='red', lw=2)

        linspace = np.linspace(np.min(param_values), np.max(param_values), 1000)
        ax3.plot(linspace, prior.distributions[i].pdf(linspace), color='green')

        plt.show()


if __name__ == '__main__':
    main()
