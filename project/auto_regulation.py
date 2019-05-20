import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from auto_regulation_routines import step_ar
from mcmc import Distribution, MetropolisHastingsABC, MetropolisHastingsPF, Prior, Proposal
from utils import check_random_state

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', choices=('abcmh', 'pmh'), required=True,
                    help='The algorithm to run. Either ABC Metropolis-Hastings or Particle Metropolis-Hastings.')
parser.add_argument('--n-samples', type=int, required=True, help='Number of Metropolis-Hastings samples.')
parser.add_argument('--n-particles', type=int, required=True, help='Number of particles.')
parser.add_argument('--burn-in', type=int, default=0, help='Length of the burn-in period.')
parser.add_argument('--thinning', type=int, default=1, help='Thinning of the Markov chain.')
parser.add_argument('--alpha', type=float, default=0.9,
                    help='Fraction of pseudo-observations covered by the kernel. Only applies if algorithm=abcmh.')
parser.add_argument('--hpr-p', type=float, default=0.95,
                    help='Width of the p-HPR of the kernel. Only applies if algorithm=abcmh.')
parser.add_argument('--kernel', choices=('gaussian', 'cauchy', 'uniform'), default='gaussian', help='Kernel type.')
args = parser.parse_args()


class ABCAutoRegulation(MetropolisHastingsABC):
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        return step_ar(x, self.const['times'][t - 1], self.const['times'][t] - self.const['times'][t - 1],
                       theta, self.const['k'], self.const['c5'], self.const['c6'])

    def _measurement_model(self, x: np.ndarray, theta: np.ndarray) -> np.array:
        out = x[:, 1] + 2 * x[:, 2]
        assert out.ndim == 1 and out.shape[0] == self.n_particles
        return out[:, np.newaxis]


class ParticleAutoRegulation(MetropolisHastingsPF):
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        return step_ar(x, self.const['times'][t - 1], self.const['times'][t] - self.const['times'][t - 1],
                       theta, self.const['k'], self.const['c5'], self.const['c6'])

    def _observation_log_prob(self, y: np.ndarray, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        log_prob = stats.norm.logpdf(y, x[:, 1] + 2 * x[:, 2], self.const['observation_std'])
        assert log_prob.ndim == 1 and log_prob.shape[0] == self.n_particles
        return log_prob


def load_data(path):
    with open(path, 'rb') as f:
        t, y = pickle.load(f)

    y += stats.norm.rvs(loc=0.0, scale=2.0, size=y.shape, random_state=1)

    return t[:250], y[1:250].reshape(-1, 1)


def main():
    algorithm = args.algorithm
    path = './auto_regulation_{}_gauss_gauss_test2'.format(algorithm)
    random_state = check_random_state(1)

    if not os.path.exists(path):
        os.makedirs(path)

    t, y = load_data('./data/ar_data.pickle')
    n_samples = args.n_samples
    n_particles = args.n_particles
    burn_in = args.burn_in
    thinning = args.thinning

    state_init = np.array([8, 8, 8, 5])

    const = {
        'k': 10,
        'c5': 0.1,
        'c6': 0.9,
        'observation_std': 2.0,
        'times': t
    }

    prior = Prior([
        # U(-7,2) as in the paper. However, in SciPy, the distribution is given as U(loc,loc+scale).
        stats.uniform(-7, 9),
        stats.uniform(-7, 9),
        stats.uniform(-7, 9),
        stats.uniform(-7, 9),
        stats.uniform(-7, 9),
        stats.uniform(-7, 9)
    ])

    proposal = Proposal([
        Distribution(stats.norm, scale=0.08),
        Distribution(stats.norm, scale=0.08),
        Distribution(stats.norm, scale=0.08),
        Distribution(stats.norm, scale=0.08),
        Distribution(stats.norm, scale=0.08),
        Distribution(stats.norm, scale=0.08)
    ])

    theta_init = np.log(np.array([0.1, 0.7, 0.35, 0.2, 0.3, 0.1]))  # stats.uniform.rvs(loc=-7, scale=9, size=6)
    random_state = check_random_state(1)

    if algorithm == 'abcmh':
        alpha = args.alpha
        hpr_p = args.hpr_p
        kernel = args.kernel

        mcmc = ABCAutoRegulation(
            n_samples=n_samples,
            n_particles=n_particles,
            alpha=alpha,
            hpr_p=hpr_p,
            state_init=state_init,
            const=const,
            kernel=kernel,
            prior=prior,
            proposal=proposal,
            theta_init=theta_init,
            random_state=random_state
        )
    else:
        mcmc = ParticleAutoRegulation(
            n_samples=n_samples,
            n_particles=n_particles,
            state_init=state_init,
            const=const,
            prior=prior,
            proposal=proposal,
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

    theta_true = np.log(np.array([0.1, 0.7, 0.35, 0.2, 0.3, 0.1]))
    theta = np.exp(theta)
    theta = theta[burn_in::thinning]
    truth = np.exp(theta_true)
    pretty_names = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', r'$c_7$', r'$c_8$']

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
