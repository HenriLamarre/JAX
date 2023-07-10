"""Farhang.py  Contains the Farhang energy optimization avalanche model

Created 2021-10-25"""
__author__ = "Henri Lamarre"

import numpy as np
import time
import scipy
import copy
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp


class Sandpile:
    """ Avalanche model using the minimal energy principle """

    def __init__(self, n):
        self.n = n
        self.B = np.zeros((self.n, self.n))
        self.C = np.zeros((self.n, self.n))
        self.curv = np.zeros((self.n, self.n))
        self.randoms = np.zeros((self.n, self.n))
        self.Zc = 1
        self.eps=1e-5

    def loop(self, maxiter):
        maxiter = int(maxiter)
        rel_e = np.zeros(maxiter)
        lat_e = np.zeros(maxiter)
        iter = 0
        while iter < range(maxiter):
            self.C.fill(0)
            self.B[:, 0] = 0
            self.B[:, -1] = 0
            self.B[0] = 0
            self.B[-1] = 0
            self.curv[1:-1, 1:-1] = self.lat_B[1:-1, 1:-1] - 1 / 4 * (self.lat_B[1:-1, 0:-2] + self.lat_B[1:-1, 2:] +
                                                                self.lat_B[0:-2, 1:-1] + self.lat_B[2:, 1:-1])
            unstable = np.where(np.abs(self.curv) > 1)
            curv_sign = np.sign(self.curv)
            if len(np.flatten(unstable))>0:
                self.randoms = np.random.uniform(0,1,size=(self.n, self.n))
                self.sums = np.roll(self.randoms,1,axis=0)+np.roll(self.randoms,-1,axis=0)+\
                            np.roll(self.randoms,1,axis=1)+np.roll(self.randoms,-1,axis=1)
                self.B -= np.multiply(curv_sign,unstable)*4*Zc/5
                self.B += np.roll(np.multiply(np.multiply(curv_sign,unstable),
                                            np.divide(np.roll(self.randoms, 1, axis=0),self.sums)), 1, axis=0)*1*Zc/5
                self.B += np.roll(np.multiply(np.multiply(curv_sign,unstable),
                                            np.divide(np.roll(self.randoms, -1, axis=0),self.sums)), -1, axis=0)*1*Zc/5
                self.B += np.roll(np.multiply(np.multiply(curv_sign,unstable),
                                            np.divide(np.roll(self.randoms, 1, axis=1),self.sums)), 1, axis=1)*1*Zc/5
                self.B += np.roll(np.multiply(np.multiply(curv_sign,unstable),
                                            np.divide(np.roll(self.randoms, -1, axis=1),self.sums)), -1, axis=1)*1*Zc/5
                lat_e[iter] = self.e_total(self.B)
                rel_e[iter] = np.roll(lat_e[iter], -1) - lat_e[iter]
                iter += 1
            else:
                niter_todo = int(np.log10(self.Zc/np.max(np.abs(self.curv)))/np.log10(1+self.eps))+1
                if niter_todo + iter > maxiter:
                    niter_todo = maxiter - iter
                rel_e[iter:iter + niter_todo] = 0
                energies = np.multiply(np.full(maxiter, lat_e[iter]), 
                                       np.power(np.full(maxiter, 1+self.eps),(np.arange(maxiter)-iter)))
                lat_e[iter:iter+niter_todo] = energies[iter:iter+niter_todo]
                iter += niter_todo

    def e_total(self, lattice):
        """ Returns the total energy of a specified lattice """
        return np.sum(np.multiply(lattice, lattice))


if __name__ == '__main__':
    a1 = Sandpile(2, 48)
    t_ = 1e4
    avalanche1.loop(t_, save =  '/home/hlamarre/PycharmProjects/Avalanches/Saves/',
                load = '/home/hlamarre/PycharmProjects/Avalanches/Saves/', progressbar=True, save_lattice=True)
    ax1 = plt.subplot(211)
    ax1.plot(avalanche1.energy_disp)
    ax1.set_ylabel('er')
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(avalanche1.energy_lat)
    ax2.set_ylabel('el')
    plt.savefig('energies.png')
    plt.show()
    #np.savez('Deterministic_lattice_history.npz', lat_history = avalanche1.lattice_history)
    #np.savez('N{}_{}_data.npz'.format(avalanche1.n, avalanche1.name), el = avalanche1.energy_lat,
             #er = avalanche1.energy_disp, ec = avalanche1.altenergy)
