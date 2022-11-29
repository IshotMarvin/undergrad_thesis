import numpy as np
import numpy.linalg as nplin
import scipy.sparse as sps
import scipy.sparse.linalg as spslin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import time
import copy

def EntEnt(vec,which_sub_sys='A',n_A=1):
    '''
    Calculates the Von Neumann entropy
    of entanglement for a pure system
    defined by vec
    Args:
        vec (list, np array, sparse array):
            state vector for the system
        which_sub_sys (string): which subsystem
            we want to find entanglement entropy for.
            Options are 'A' and 'B'; default is 'A'
        n_A (int): number of particles in
            subsystem A; default is 1
    Returns:
        (float) entanglement entropy of
            subsystem A in vec state
    '''
    n_B = n - n_A
    d_A = 2**n_A
    d_B = 2**n_B

    basis = np.identity(d_B)
    basis_full = []
    for i in range(d_B):
        basis_full.append(np.kron(np.identity(d_A),basis[:][i]))


    rho = np.outer(vec,vec)
    rho_reduced = np.zeros((d_A,d_A))
    for i in range(len(basis_full)):
        rho_reduced += np.conj(basis_full[i]).dot(rho).dot(np.transpose(basis_full[i]))
  


vec = np.array([0,1/np.sqrt(2),1/np.sqrt(2),0])
rho = np.outer(vec,vec)
print(rho)

blah = vec.dot(rho).dot(vec)
print(blah)

print(np.log(rho))
print(np.trace(np.log(rho)))
