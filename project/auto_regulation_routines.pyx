import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from libc.stdlib cimport rand, srand, RAND_MAX


cpdef transition(long[:,:] state, int t, theta, int n_particles, consts):
    cdef double c1 = exp(theta['lc1'])
    cdef double c2 = exp(theta['lc2'])
    cdef double c3 = exp(theta['lc3'])
    cdef double c4 = exp(theta['lc4'])
    cdef double c5 = consts['c5']
    cdef double c6 = consts['c6']
    cdef double c7 = exp(theta['lc7'])
    cdef double c8 = exp(theta['lc8'])

    cdef int k = consts['k']
    cdef int i

    cdef long[:,:] S = consts['S']
    cdef double[:] times = consts['t']
    cdef double[:] h = np.empty(shape=8)

    state = state.copy()
    state_new = np.zeros(shape=(4, n_particles), dtype=np.long)

    for i in range(n_particles):
        state_new[:, i] = _simulate_state(state[:, i], S,
                                          c1, c2, c3, c4, c5, c6, c7, c8,
                                          k, h, times[t - 1], times[t])

    return state_new


cdef _simulate_state(long[:] state, long[:,:] S,
                    double c1, double c2, double c3, double c4,
                    double c5, double c6, double c7, double c8,
                    int k, double[:] h, double t0, double T):
    cdef double t = t0
    cdef double h_sum = 0.0
    cdef double s1, s2, dt, cum_prob
    cdef int i, reaction_type
    cdef long rna, P, P2, dna

    while t < T:
        # Calculate the hazard function.
        rna = state[0]
        P = state[1]
        P2 = state[2]
        dna = state[3]

        h[0] = c1 * dna * P2
        h[1] = c2 * (k - dna)
        h[2] = c3 * dna
        h[3] = c4 * rna
        h[4] = c5 * P * (P - 1) / 2
        h[5] = c6 * P2
        h[6] = c7 * rna
        h[7] = c8 * P

        # Calculate the hazard function sum and reaction probabilities.
        h_sum = 0.0
        for i in range(8):
            h_sum += h[i]

        # Calculate time delta.
        s1 = rand()
        s1 /= RAND_MAX
        dt = -log(s1) / h_sum

        # Select reaction type.
        s2 = rand()
        s2 /= RAND_MAX
        cum_prob = 0.0

        # Handle the case when s2 == 1.0, i.e. the last reaction should be selected, but would not due to numerical issues.
        reaction_type = 7
        for i in range(8):
            cum_prob += h[i] / h_sum

            if s2 <= cum_prob:
                reaction_type = i
                break

        # Perform the selected reaction.
        for i in range(4):
            state[i] += S[i, reaction_type]

        t += dt

    return state