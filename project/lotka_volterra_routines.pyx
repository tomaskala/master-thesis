import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from libc.stdlib cimport rand, RAND_MAX


cpdef step_lv(long[:,:] x, double t0, double deltat, double[:] th):
    cdef int i

    x = x.copy()
    x_new = np.zeros(shape=(x.shape[0], x.shape[1]), dtype=np.long)

    for i in range(x.shape[0]):
        x_new[i] = simulate_state(x[i], t0, deltat, th)

    return x_new


cdef simulate_state(long[:] x, double t0, double deltat, double[:] th):
    cdef double t = t0
    cdef double dt = deltat
    cdef double termt = t + dt
    cdef double h0, h1, h2, h3, u, s

    while True:
        h1 = exp(th[0]) * x[0]
        h2 = exp(th[1]) * x[0] * x[1]
        h3 = exp(th[2]) * x[1]
        h0 = h1+h2+h3

        if h0 < 1e-10 or x[0] >= 1000000:
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
            x[0] += 1
        elif u < (h1 + h2) / h0:
            x[0] -= 1; x[1] += 1
        else:
            x[1] -= 1
