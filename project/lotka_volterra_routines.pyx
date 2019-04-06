import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from libc.stdlib cimport rand, RAND_MAX

"""
void stepLV(int *x,double *t0p,double *dtp,double *c)
{
  double t=*t0p, dt=*dtp, termt=t+dt;
  GetRNGstate();
  double h0,h1,h2,h3,u;
  while (1==1) {
    h1=c[0]*x[0]; h2=c[1]*x[0]*x[1]; h3=c[2]*x[1];
    h0=h1+h2+h3;
    if ((h0<1e-10)||(x[0]>=1000000))
      t=1e99;
    else
      t+=rexp(1.0/h0);
    if (t>=termt) {
      PutRNGstate();
      return;
    }
    u=unif_rand();
    if (u<h1/h0)
      x[0]+=1;
    else if (u<(h1+h2)/h0) {
      x[0]-=1; x[1]+=1;
    } else
      x[1]-=1;
  }
}
"""

def step_LVc(long[:,:] x, double t0, double deltat, double[:] th):
    cdef int i

    x = x.copy()
    x_new = np.zeros(shape=(x.shape[0], x.shape[1]), dtype=np.long)

    for i in range(x.shape[0]):
        x_new[i] = one_step(x[i], t0, deltat, th)

    return x_new

cpdef one_step(long[:] x, double t0, double deltat, double[:] th):
    cdef double t = t0
    cdef double dt = deltat
    cdef double termt = t + dt
    cdef double h0, h1, h2, h3, u, s

    while True:
        h1=exp(th[0])*x[0]; h2=exp(th[1])*x[0]*x[1]; h3=exp(th[2])*x[1]
        h0=h1+h2+h3

        if h0 < 1e-10 or x[0] >= 1000000:
            t=1e99
        else:
            s = rand()
            s /= RAND_MAX
            t -= log(s) / h0

        if t >= termt:
            return x

        u = rand()
        u /= RAND_MAX

        if u<  h1 / h0:
            x[0]+=1
        elif u < (h1 + h2) / h0:
            x[0]-=1; x[1]+=1
        else:
            x[1]-=1


cpdef transition(long[:,:] state, theta):
    cdef double th1 = exp(theta['lth1'])
    cdef double th2 = exp(theta['lth2'])
    cdef double th3 = exp(theta['lth3'])
    cdef int i
    state = state.copy()
    state_new = np.zeros(shape=(state.shape[0], state.shape[1]), dtype=np.long)

    for i in range(state.shape[1]):
        state_new[:, i] = _simulate_state(state[:, i], th1, th2, th3)

    return state_new


cdef _simulate_state(long[:] x, double c0, double c1, double c2):
    cdef double t = 0.0
    cdef double T = 2.0
    cdef double h0, h1, h2, h3, u, s

    while True:
        h1=c0*x[0]; h2=c1*x[0]*x[1]; h3=c2*x[1]
        h0=h1+h2+h3

        if h0 < 1e-10 or x[0] >= 1000000:
            t=1e99
        else:
            s = rand()
            s /= RAND_MAX
            t -= log(s) / h0

        if t >= T:
            return x

        u = rand()
        u /= RAND_MAX

        if u<  h1 / h0:
            x[0]+=1
        elif u < (h1 + h2) / h0:
            x[0]-=1; x[1]+=1
        else:
            x[1]-=1