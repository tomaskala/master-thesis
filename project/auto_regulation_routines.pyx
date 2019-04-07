import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from libc.stdlib cimport rand, RAND_MAX


cpdef step_ar(long[:,:] x, double t0, double deltat, double[:] th, int k, double c5, double c6):
    cdef double c1 = exp(th[0])
    cdef double c2 = exp(th[1])
    cdef double c3 = exp(th[2])
    cdef double c4 = exp(th[3])
    cdef double c7 = exp(th[4])
    cdef double c8 = exp(th[5])
    cdef int i

    x = x.copy()
    x_new = np.zeros(shape=(x.shape[0], x.shape[1]), dtype=np.long)

    for i in range(x.shape[0]):
        x_new[i] = simulate_state(x[i], t0, deltat, k, c1, c2, c3, c4, c5, c6, c7, c8)

    return x_new


cdef simulate_state(long[:] x, double t0, double deltat, int k,
                    double c1, double c2, double c3, double c4,
                    double c5, double c6, double c7, double c8):
    cdef double t = t0
    cdef double dt = deltat
    cdef double termt = t + dt
    cdef double h0, h1, h2, h3, h4, h5, h6, h7, h8, u, s

    while True:
        h1 = c1 * x[3] * x[2]
        h2 = c2 * (k - x[3])
        h3 = c3 * x[3]
        h4 = c4 * x[0]
        h5 = c5 * x[1] * (x[1] - 1) / 2
        h6 = c6 * x[2]
        h7 = c7 * x[0]
        h8 = c8 * x[1]
        h0 = h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8

        if h0 < 1e-10 or x[0] >= 1000000 or x[1] >= 1000000 or x[2] >= 1000000 or x[3] >= 1000000:
            t = 1e99
        else:
            s = rand()
            s /= RAND_MAX
            t -= log(s) / h0

        if t >= termt:
            return x

        u = rand()
        u /= RAND_MAX

        if u < h1 / h0:
            x[2] -= 1; x[3] -= 1
        elif u < (h1 + h2) / h0:
            x[2] += 1; x[3] += 1
        elif u < (h1 + h2 + h3) / h0:
            x[0] += 1
        elif u < (h1 + h2 + h3 + h4) / h0:
            x[1] += 1
        elif u < (h1 + h2 + h3 + h4 + h5) / h0:
            x[1] -= 2; x[2] += 1
        elif u < (h1 + h2 + h3 + h4 + h5 + h6) / h0:
            x[1] += 2; x[2] -= 1
        elif u < (h1 + h2 + h3 + h4 + h5 + h6 + h7) / h0:
            x[0] -= 1
        else:
            x[1] -= 1
