import os
import pickle
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from auto_regulation_routines import step_ar
from mcmc import Distribution, MetropolisHastingsABC, MetropolisHastingsPF, Prior, Proposal
from utils import check_random_state


class ABCAutoRegulation(MetropolisHastingsABC):
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        return step_ar(x, self.const['times'][t - 1], self.const['times'][t] - self.const['times'][t - 1], theta,
                       self.const['k'], self.const['c5'], self.const['c6'])

    def _measurement_model(self, x: np.ndarray, theta: np.ndarray) -> np.array:
        return x[:, 1] + 2 * x[:, 2]


class ParticleAutoRegulation(MetropolisHastingsPF):
    def _transition(self, x: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        return step_ar(x, self.const['times'][t - 1], self.const['times'][t] - self.const['times'][t - 1], theta,
                       self.const['k'], self.const['c5'], self.const['c6'])

    def _observation_log_prob(self, y: np.ndarray, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        log_prob = stats.norm.logpdf(y, x[:, 1] + 2 * x[:, 2], self.const['observation_std'])
        assert log_prob.ndim == 1 and log_prob.shape[0] == self.n_particles
        return log_prob


def hazard_function(state: np.ndarray, theta: np.ndarray, const: Dict[str, float]):
    c1 = np.exp(theta[0])
    c2 = np.exp(theta[1])
    c3 = np.exp(theta[2])
    c4 = np.exp(theta[3])
    c5 = const['c5']
    c6 = const['c6']
    c7 = np.exp(theta[4])
    c8 = np.exp(theta[5])

    rna = state[0]
    p = state[1]
    p2 = state[2]
    dna = state[3]

    return np.array([
        c1 * dna * p2,
        c2 * (const['k'] - dna),
        c3 * dna,
        c4 * rna,
        c5 * p * (p - 1) / 2,
        c6 * p2,
        c7 * rna,
        c8 * p
    ])


def simulate_state(x: np.ndarray, t0: float, deltat: float, theta: np.ndarray, const: Dict[str, float], random_state):
    t = t0
    dt = deltat
    termt = t + dt

    while True:
        h = hazard_function(x, theta, const)
        h0 = np.sum(h)

        if h0 < 1e-10 or np.any(x >= 1000000):
            t = 1e99
        else:
            s = random_state.rand()
            t -= np.log(s) / h0

        if t >= termt:
            return x, t

        u = random_state.rand()

        if u < h[0] / h0:
            x[2] -= 1
            x[3] -= 1
        elif u < (h[0] + h[1]) / h0:
            x[2] += 1
            x[3] += 1
        elif u < (h[0] + h[1] + h[2]) / h0:
            x[0] += 1
        elif u < (h[0] + h[1] + h[2] + h[3]) / h0:
            x[1] += 1
        elif u < (h[0] + h[1] + h[2] + h[3] + h[4]) / h0:
            x[1] -= 2
            x[2] += 1
        elif u < (h[0] + h[1] + h[2] + h[3] + h[4] + h[5]) / h0:
            x[1] += 2
            x[2] -= 1
        elif u < (h[0] + h[1] + h[2] + h[3] + h[4] + h[5] + h[6]) / h0:
            x[0] -= 1
        else:
            x[1] -= 1


"""
gillespie <- function(N, n, ...)
{
tt = 0
x = N$M
S = t(N$Post-N$Pre)
u = nrow(S)
v = ncol(S)
tvec = vector("numeric",n)
xmat = matrix(ncol=u,nrow=n+1)
xmat[1,] = x
for (i in 1:n) {
h = N$h(x,tt, ...)
tt = tt+rexp(1,sum(h))
j = sample(v,1,prob=h)
x = x+S[,j]
tvec[i] = tt
xmat[i+1,] = x
}
return(list(t=tvec, x=xmat))
}"""


# Gillespie algorithm.
def simulate_xy(path: str, T: int, theta: np.ndarray, const: Dict[str, float], random_state=None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    if os.path.exists(path):
        with open(path, mode='rb') as f:
            return pickle.load(f)
    else:
        tt = 0
        x = np.array([8, 8, 8, 5])
        S = const['S']
        u, v = S.shape

        xmat = np.zeros(shape=(T + 1, u), dtype=int)
        tvec = np.zeros(shape=T + 1, dtype=float)
        xmat[0] = x

        for i in range(T):
            h = hazard_function(x, theta, const)
            tt += random_state.exponential(scale=1 / np.sum(h))
            j = random_state.choice(v, p=h / np.sum(h))
            x += S[:, j]
            tvec[i + 1] = tt
            xmat[i + 1] += x

        ymat = random_state.normal(loc=xmat[1:, 1] + 2 * xmat[1:, 2], scale=const['observation_std'])

        assert xmat.shape[0] == ymat.shape[0] + 1 == tvec.shape[0]

        with open(path, mode='wb') as f:
            pickle.dump((tvec, xmat, ymat), f)

        return tvec, xmat, ymat


# Gillespie algorithm.
def simulate_xy2(path: str, T: int, theta: np.ndarray, const: Dict[str, float], random_state=None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    if os.path.exists(path):
        with open(path, mode='rb') as f:
            return pickle.load(f)
    else:
        x_0 = np.array([8, 8, 8, 5])
        x = [x_0]
        y = []

        x_prev = x_0
        t = 0.0
        time = [t]

        while t < T:
            h = hazard_function(x_prev, theta, const)
            h_sum = np.sum(h)
            reaction_type = random_state.choice(h.shape[0], p=h / h_sum)
            dt = -np.log(random_state.rand()) / h_sum

            x_next = x_prev + const['S'][:, reaction_type]
            t += dt
            time.append(t)

            # Update the state.
            x.append(x_next)
            x_prev = x_next

            # Generate observation.
            y_next = x_next[1] + 2 * x_next[2] + stats.norm.rvs(loc=0.0, scale=const['observation_std'],
                                                                random_state=random_state)
            y.append(y_next)

        x = np.array(x)
        y = np.array(y)
        time = np.array(time)
        assert x.shape[0] == time.shape[0] == y.shape[0] + 1

        with open(path, mode='wb') as f:
            pickle.dump((time, x, y), f)

        return time, x, y


def load_data(path):
    with open(path, 'rb') as f:
        t, y = pickle.load(f)

    return t, y


# TODO: Simulate in R as in Wilkinson's book (around p. 189) and then import it here?
def main():
    algorithm = 'pmh'
    path = './auto_regulation_{}'.format(algorithm)
    random_state = check_random_state(1)

    if not os.path.exists(path):
        os.makedirs(path)

    n_samples = 2000
    n_particles = 200
    thinning = 10

    state_init = np.array([8, 8, 8, 5])

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

    theta_init = np.log(np.array([0.1, 0.7, 0.35, 0.2, 0.3, 0.1]))
    random_state = check_random_state(1)

    # t, x, y = simulate_xy(os.path.join(path, 'simulated_data.pickle'), T=200, theta=theta_init,
    #                       const=const, random_state=random_state)
    t, y = load_data('./data/auto_regulation.pickle')

    # print(t)
    # print(y)
    # print(t.shape)
    # print(y.shape)
    # exit()

    const['times'] = np.concatenate(([0.0], t))

    if algorithm == 'abcmh':
        alpha = 0.9
        hpr_p = 0.95
        kernel = 'gaussian'

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
            tune=False,
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
    truth = np.exp(theta_init)
    pretty_names = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', r'$c_7$', r'$c_8']

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
