import numbers
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous


def check_random_state(random_state) -> np.random.RandomState:
    if random_state is None or random_state is np.random:
        return np.random.mtrand._rand
    if isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState instance.'.format(random_state))


def plot_parameters(thetas: List[Dict[str, float]],
                    pretty_names: Optional[Dict[str, str]] = None,
                    true_values: Optional[Dict[str, float]] = None,
                    priors: Optional[Dict[str, rv_continuous]] = None,
                    paths: Optional[Dict[str, str]] = None):
    """
    Plot the histograms of the sampled theta, representing the estimate of p(theta|y),
    possibly along with some additional quantities.
    :param thetas: list of sampled parameters
    :param pretty_names: optional mapping from variable names to how they should be shown in the plot title
    :param true_values: optional mapping from variable names to their true values, will be shown in the plots
    :param priors: optional mapping from variable names to the prior distributions, will plot the densities
    :param paths: optional mapping from variable names to paths where the plots should be stored, instead of shown
    """
    assert len(thetas) > 0
    params2samples = {param_name: [] for param_name in thetas[0].keys()}

    for theta in thetas:
        for param_name, param_value in theta.items():
            params2samples[param_name].append(param_value)

    fig = plt.figure()

    for param_name, param_values in params2samples.items():
        show_prior = priors is not None and param_name in priors
        plt.hist(param_values, density=show_prior)

        if show_prior:
            x = np.linspace(np.min(param_values), np.max(param_values), 100)
            plt.plot(x, priors[param_name].pdf(x), color='green')

        if pretty_names is not None and param_name in pretty_names:
            pretty_name = pretty_names[param_name]
        else:
            pretty_name = param_name

        title = '{}, mean: {:.03f}'.format(pretty_name, np.mean(param_values))

        if true_values is not None and param_name in true_values:
            plt.axvline(true_values[param_name], color='red', lw=2)
            title += ', true value: {:.03f}'.format(true_values[param_name])

        plt.title(title)

        if paths is not None and param_name in paths:
            fig.savefig(paths[param_name])
        else:
            plt.show()
