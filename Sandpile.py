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
from functools import partial
import copy


class Sandpile:
    """Deterministic Conservative Avalanche model
     with anisotropic redistribution"""

    def __init__(self, n, eps=1e-5):
        self.n = n # Width of the lattice
        self.B = jnp.zeros((self.n, self.n)) #The lattice
        self.Zc = 1 # The curvature criterion
        self.eps=eps # The driving parameter
        self.rng = np.random.default_rng(seed=1) # The randomness seed
        self.e0 = 4/5*self.Zc**2 # The energy quantum


    def loop_(self, maxiter, verbose=False):
        """Function in the class to run the model over maxiter iterations.
        The function loop and every other function called in it are outside the class
        as Jax cant precompile methods with self in the argument."""
        rel_e, lat_e = loop(self.B, self.e0, self.Zc, self.eps,
                             self.rng, self.n, maxiter, verbose)
        return rel_e, lat_e # Returns released energy and lattice energy
    
def loop(B, e0, Zc, eps, rng, n, maxiter, verbose):
    maxiter = int(maxiter) # The max number of iterations
    B = jnp.array(B) # We call this in case we set B after __init__ is called
    rel_e = jnp.zeros(maxiter+1, dtype=float)# Jax requires that every array has fixed shape 
    lat_e = jnp.zeros(maxiter+1, dtype=float)
    lat_e = lat_e.at[0].set(e_total(B, e0))# initialize the lattice energy as we compute the
    rel_e = lat_e.at[0].set(0)# released energy using differences. 
    # These first entry are discarded at the end
    iter = jnp.array([1]) # The iteration variable. 
    #It needs to be a Jax object as some functions use it as an argument
    start_loop = time.time()
    while iter[0] < maxiter+1: # The loop
        B, iter, lat_e, rel_e = in_loop(B, lat_e, rel_e, iter,
                                         rng,n,Zc, e0,eps, maxiter)
    if verbose:
        print('The loop took {} seconds'.format(round(time.time()-start_loop,2)))
    # The following part sets the energy for the driving as I did not know how to
    # do it inside a jit-compiled function. This could probably be done better
    iszero = np.concatenate(([0], np.equal(lat_e, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    for range_ in ranges:
        length = range_[1]-range_[0]
        initEnj = lat_e[range_[0]-1]
        mult = np.power(np.full(length, (1+eps)),2*np.arange(length))
        vals = np.multiply(np.full(length, initEnj), mult)
        lat_e = lat_e.at[range_[0]:range_[1]].set(vals)
    rel_e = rel_e.at[rel_e<0].set(e0) # Some combinations of rns yield negative energy release
    # We fix it at e0 as it is rare.
    return rel_e[1:], lat_e[1:]
    

@partial(jit, static_argnums=(4,5,6,7,8,9,))
def in_loop(B,lat_e, rel_e, iter, rng,n,Zc,e0,eps, maxiter):
    """ The loop 
    Jax cant compile functions with if statements that depend
    on the arguments so for every loop, we do both the redistribution 
    and the driving. When the model is unstable, the redistribution takes
    place and we drive for 0 iterations. When the model is stable, we 
    redistribute 0 nodes and drive for x iterations."""
    curv, unstable, curv_sign = curv_calc(B, n, Zc) # Computes the curvature
    B = B.at[:,0].set(0) # Flushes nodal value at the edges
    B = B.at[:,-1].set(0)
    B = B.at[0].set(0)
    B = B.at[-1].set(0)
    B, lat_e, rel_e, iter = redis(B, curv_sign, unstable,
                                        lat_e, rel_e, iter, rng, 
                                        n, Zc, e0) # Redistribution
    # Computes how many iterations of driving need to happen
    niter_todo = jnp.log10(Zc/jnp.max(jnp.abs(curv)))/jnp.log10(1+eps)+1
    niter_todo = niter_todo.astype(int)
    niter_todo = jnp.max(jnp.array([niter_todo, 0])) # if model is unstable
    # niter_todo is 0
    niter_todo = jnp.min(jnp.array([niter_todo, maxiter+1 - iter[0]]))
    # If niter_todo is larger than maxiter, we cap it at maxiter
    B *= ((1+eps)**niter_todo)# Drive the model
    lat_e = lat_e.at[iter[0]+niter_todo].set(e_total(B, e0)) # compute the
    # new lattice energy
    iter = iter.at[0].set(iter[0]+niter_todo) # update the iteration variable
    return B, iter, lat_e, rel_e
    

@partial(jit, static_argnums=(1,))
def e_total(lattice, e0):
    """ Returns the total energy of a specified lattice """
    return jnp.sum(jnp.multiply(lattice, lattice))/e0


@partial(jit, static_argnums=(1,2,))
def curv_calc(B,n,Zc):
    """ Computes the curvature of the model, which nodes are unstable and
    the sign of the curvature for the increments in redistribution"""
    curv = jnp.zeros((n, n)) 
    curv = curv.at[1:-1, 1:-1].set(B[1:-1, 1:-1] - 1 / 4 * (B[1:-1, 0:-2] + B[1:-1, 2:] +
                                                        B[0:-2, 1:-1] + B[2:, 1:-1]))
    unstable = jnp.where(jnp.abs(curv) > Zc, 1, 0)
    curv_sign = jnp.sign(curv)
    return curv, unstable, curv_sign

@partial(jit, static_argnums=(6,7,8,9,))
def redis(B, curv_sign, unstable, lat_e, rel_e, iter, rng, n, Zc, e0):
    """ Redistributes the unstable nodes to their neighbours """
    randoms = rng.uniform(0,1,size=(n, n)) #Computes a random number between 0 and 1 
    # at every position in the lattice
    sums = jnp.roll(randoms,1,axis=0)+jnp.roll(randoms,-1,axis=0)+\
                jnp.roll(randoms,1,axis=1)+jnp.roll(randoms,-1,axis=1)
    # Sums the four neihgbour random numbers for every position in the lattice
    B -= jnp.multiply(curv_sign,unstable)*4*Zc/5 # Redistribute
    B += jnp.roll(jnp.multiply(jnp.multiply(curv_sign,unstable),
                                jnp.divide(jnp.roll(randoms, -1, axis=0),sums)), 1, axis=0)*4*Zc/5
    B += jnp.roll(jnp.multiply(jnp.multiply(curv_sign,unstable),
                                jnp.divide(jnp.roll(randoms, 1, axis=0),sums)), -1, axis=0)*4*Zc/5
    B += jnp.roll(jnp.multiply(jnp.multiply(curv_sign,unstable),
                                jnp.divide(jnp.roll(randoms, -1, axis=1),sums)), 1, axis=1)*4*Zc/5
    B += jnp.roll(jnp.multiply(jnp.multiply(curv_sign,unstable),
                                jnp.divide(jnp.roll(randoms, 1, axis=1),sums)), -1, axis=1)*4*Zc/5
    lat_e = lat_e.at[iter[0]].set(e_total(B, e0)) #Computes the lattice energy
    rel_e = rel_e.at[iter[0]].set(lat_e[iter[0]-1] - lat_e[iter[0]]) # Computes the released energy
    iter = iter.at[0].set(iter[0]+1) # Updates the iteration parameter
    return B, lat_e, rel_e, iter