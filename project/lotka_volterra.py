import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyximport;

pyximport.install(setup_args={'include_dirs': np.get_include()})

from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from lotka_volterra_routines import step_lv
from mcmc import Distribution, MetropolisHastingsPF, Prior, Proposal
from utils import check_random_state


class ParticleLotkaVolterra(MetropolisHastingsPF):
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        return step_lv(x, self.const['times'][t - 1], self.const['times'][t] - self.const['times'][t - 1], theta)

    def _observation_log_prob(self, y: np.ndarray, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        log_prob = np.sum(stats.norm.logpdf(y, x, self.const['observation_std']), axis=1)
        assert log_prob.ndim == 1 and log_prob.shape[0] == self.n_particles
        return log_prob


def load_data(path):
    with open(path, 'rb') as f:
        t, y = pickle.load(f)

    return t, y


def main():
    algorithm = 'pmh'
    path = './lotka_volterra_{}'.format(algorithm)
    random_state = check_random_state(1)

    if not os.path.exists(path):
        os.makedirs(path)

    t, y = load_data('./data/LV_data.pickle')
    n_samples = 10000
    n_particles = 100
    thinning = 10

    state_init = np.c_[
        stats.poisson.rvs(mu=50, size=n_particles, random_state=random_state),
        stats.poisson.rvs(mu=100, size=n_particles, random_state=random_state)
    ]

    const = {
        'observation_std': 10.0,
        'times': np.concatenate(([0.0], t))
    }

    prior = Prior([
        stats.uniform(-50, 100),
        stats.uniform(-50, 100),
        stats.uniform(-50, 100)
    ])

    proposal = Proposal([
        Distribution(stats.norm, scale=0.01),
        Distribution(stats.norm, scale=0.01),
        Distribution(stats.norm, scale=0.01)
    ])

    theta_init = np.log(np.array([1, 0.005, 0.6]))

    if algorithm == 'abcmh':
        raise NotImplementedError()
    else:
        mcmc = ParticleLotkaVolterra(
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

    theta = np.exp(theta)
    theta = theta[::thinning]
    truth = np.array([1, 0.005, 0.6])
    pretty_names = [r'$c_1$', r'$c_2$', r'$c_3$']

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

        plt.show()


if __name__ == '__main__':
    main()
