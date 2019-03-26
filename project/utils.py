import numbers
from statsmodels.graphics.tsaplots import plot_acf
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_continuous


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
                    kernel_scales: Optional[List[float]] = None,
                    paths: Optional[Dict[str, str]] = None,
                    bins: Optional[int] = None,
                    burn_in: int = 0,
                    step: int = 1,
                    max_lags: int = 10):
    """
    Plot the histograms of the sampled theta, representing the estimate of p(theta|y),
    possibly along with some additional quantities.
    :param thetas: list of sampled parameters
    :param pretty_names: optional mapping from variable names to how they should be shown in the plot title
    :param true_values: optional mapping from variable names to their true values, will be shown in the plots
    :param priors: optional mapping from variable names to the prior distributions, will plot the densities
    :param kernel_scales: optional list of scales of the ABC kernel
    :param paths: optional mapping from variable names to paths where the plots should be stored, instead of shown
    :param bins: optional number of bins for the histogram
    :param burn_in: drop this many samples from the beginning
    :param step: only show every `step`th sample, in the hope of removing the correlation between consecutive samples
    :param max_lags: maximum number of lags to show in the autocorrelation plot
    """
    assert len(thetas) > 0
    params2samples = {param_name: [] for param_name in thetas[0].keys()}

    for theta in thetas:
        for param_name, param_value in theta.items():
            params2samples[param_name].append(param_value)

    for param_name, param_values in params2samples.items():
        if kernel_scales is None or len(kernel_scales) <= 1:
            # Either the kernel scales have not been given or just the initial scale is present.
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

            ax4.set_title('Scales of the ABC kernel')
            ax4.plot(kernel_scales)

        # Trace plot.
        ax1.set_title('Trace plot')
        ax1.plot(param_values[burn_in::step])

        # Autocorrelation.
        plot_acf(param_values[burn_in::step], ax=ax2, lags=max_lags)

        # Histogram.
        ax3.set_title('Histogram')

        show_prior = priors is not None and param_name in priors
        ax3.hist(param_values[burn_in::step], density=show_prior, bins=bins)

        if show_prior:
            x = np.linspace(np.min(param_values[burn_in::step]), np.max(param_values[burn_in::step]), 100)
            ax3.plot(x, priors[param_name].pdf(x), color='green')

        if pretty_names is not None and param_name in pretty_names:
            pretty_name = pretty_names[param_name]
        else:
            pretty_name = param_name

        title = '{}, mean: {:.03f}'.format(pretty_name, np.mean(param_values[burn_in::step]))

        if true_values is not None and param_name in true_values:
            ax3.axvline(true_values[param_name], color='red', lw=2)
            title += ', true value: {:.03f}'.format(true_values[param_name])

            plt.suptitle(title)

        if paths is not None and param_name in paths:
            fig.savefig(paths[param_name])
        else:
            plt.show()
