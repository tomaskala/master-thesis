import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


matplotlib.rc('font', size=32)
lw = 6
s = 200


def kernel_comparison():
    fig = plt.figure()

    loc = 0.0
    scale = 3.0

    x = np.linspace(-20, 20, 1000)
    plt.plot(x, stats.norm.pdf(x, loc=loc, scale=scale), lw=lw, color='dimgrey', label='Gaussian')
    plt.plot(x, stats.cauchy.pdf(x, loc=loc, scale=scale), lw=lw, color='black', label='Cauchy')
    plt.plot(x, stats.uniform.pdf(x, loc=loc-scale, scale=loc+2*scale), lw=lw, color='crimson', label='Uniform')

    plt.legend()
    plt.title('Kernel comparison')
    plt.xticks([loc], ['y'])
    plt.yticks([])
    plt.show()


def kernel_tuning():
    y = 5.0
    u = np.array([-1.0, 0.0, 2.0, 6.0, 8.0, 10.0])
    u_alpha = 6.0 + 1

    p = 0.95
    eps = abs(u_alpha - y) / stats.norm.ppf((1+p)/2)

    fig = plt.figure()

    x = np.linspace(-5, 15, 1000)
    plt.plot(x, stats.norm.pdf(x, loc=y, scale=eps), lw=lw, color='black')

    plt.vlines(y, ymin=0.0, ymax=stats.norm.pdf(y, loc=y, scale=eps), lw=lw, color='crimson')
    plt.vlines(u_alpha, ymin=0.0, ymax=stats.norm.pdf(u_alpha, loc=y, scale=eps), lw=lw, color='crimson')
    plt.scatter(u, np.zeros(u.size), s=s, color='dimgrey', zorder=1000)
    plt.scatter(u_alpha, 0, s=s, color='crimson', zorder=1000)
    plt.scatter(y, 0, s=s, color='crimson', zorder=1000)

    shaded_x = x[(x >= y) & (x <= u_alpha)]
    plt.fill_between(shaded_x, stats.norm.pdf(shaded_x, loc=y, scale=eps), color='black', alpha=0.4, linewidth=0)

    plt.xticks([y, u_alpha], ['$y_t$', r'$u_t^{[\alpha]}$'])
    plt.yticks([])
    plt.title('Kernel tuning')
    plt.show()



if __name__ == '__main__':
    #kernel_comparison()
    kernel_tuning()

