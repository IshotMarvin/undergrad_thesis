import numpy as np
import numpy.linalg as nplin
import scipy.sparse as sps
import scipy.sparse.linalg as spslin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import seaborn as sns
import time
import copy

n = int(input("Enter number of spins: "))

start = time.time()

## ## Form the basis

basis = sps.identity(2**n)

## ## Find the matrix representation of the spin operators for each particle

def S_plus(m):
    '''
    Produces the matrix representation of 
    the S_plus operator, which raises a 
    spin down state to up, or annihilates
    the state if it's already spin up.
    Args:
        m: (integer) which particle the 
            operator is acting on
    Returns:
	(sparse array) matrix representation
       	    of the S_plus operator acting
	    on the given particle
    '''
    S_plus = sps.eye(2,k=1)
    S_plus = sps.kron(sps.identity(2**m),S_plus)
    S_plus = sps.kron(S_plus,sps.identity(2**(n-m-1)))
    return S_plus

def S_minus(m):
    '''
    Produces the matrix representation of 
    the S_minus operator, which lowers a 
    spin up state to down, or annihilates
    the state if it's already spin down.
    Args:
        m: (integer) which particle the 
	    operator is acting on
    Returns:
	(sparse array) matrix representation
            of the S_plus operator acting
	    on the given particle
    '''
    S_minus = sps.eye(2,k=-1)
    S_minus = sps.kron(np.identity(2**m),S_minus)
    S_minus = sps.kron(S_minus,sps.identity(2**(n-m-1)))
    return S_minus

def S_z(m):
    '''
    Produces the matrix representation of
    the S_z operator, which finds the value
    of spin for the given particle in the z
    direction.
    Args:
        m: (integer) which particle the 
            operator is acting on
    Returns:
        (sparse array) matrix representation 
        of the S_z operator acting on the
        given particle
    '''
    S_z = sps.diags([1,-1])
    S_z = sps.kron(sps.identity(2**m),S_z)
    S_z = sps.kron(S_z,sps.identity(2**(n-m-1)))
    return 1/2*S_z

def S_x(m):
    '''
    Produces the matrix representation of
    the S_x operator, which finds the value
    of spin for the given particle in the x
    direction.
    Args:
        m: (integer) which particle the 
            operator is acting on
    Returns:
        (sparse array) matrix representation 
        of the S_x operator acting on the
        given particle
    '''
    S_x = np.zeros([2,2])
    S_x[0,1] = 1
    S_x[1,0] = 1
    S_x = sps.csr_matrix(S_x)
    S_x = sps.kron(sps.identity(2**(m)),S_x)
    S_x = sps.kron(S_x,sps.identity(2**(n-m-1)))
    return 1/2*S_x

def S_y(m):
    '''
    Produces the matrix representation of
    the S_y operator, which finds the value
    of spin for the given particle in the y
    direction.
    Args:
        m: (integer) which particle the 
            operator is acting on
    Returns:
        (sparse array) matrix representation 
        of the S_y operator acting on the
        given particle
    '''
    S_y = np.zeros([2,2],dtype=complex)
    S_y[0,1] = -1j
    S_y[1,0] = 1j
    S_y = sps.csr_matrix(S_y)
    S_y = sps.kron(sps.identity(2**(m)),S_y)
    S_y = sps.kron(S_y,sps.identity(2**(n-m-1)))
    return 1/2*S_y

op_dict = {}
for i in range(n):
    op_dict[(i,1)] = S_x(i)
    op_dict[(i,2)] = S_y(i)
    op_dict[(i,3)] = S_z(i)

## ## Form the Hamiltonian: Chiral Interaction with Nearest Neighbor Coupling

eigValList1 = []
eigList1 = []
eigValList2 = []
eigList2 = []
eigValList3 = []
eigList3 = []
eigValList4 = []
eigList4 = []
eigValList5 = []
eigList5 = []
vecList = []
JList = []
J = -1
gsList = []

## Nearest Neighbor (Heisenberg) coupling: S dot S_neighbors

H_neighbor = (op_dict[(n-2,1)]*op_dict[(n-1,1)] +
              op_dict[(n-2,2)]*op_dict[(n-1,2)] +
              op_dict[(n-2,3)]*op_dict[(n-1,3)])

for i in range(n-2):
    H_neighbor += (op_dict[(i,1)]*op_dict[(i+1,1)] +
                   op_dict[(i,2)]*op_dict[(i+1,2)] +
                   op_dict[(i,3)]*op_dict[(i+1,3)] +
                   op_dict[(i,1)]*op_dict[(i+2,1)] +
                   op_dict[(i,2)]*op_dict[(i+2,2)] +
                   op_dict[(i,3)]*op_dict[(i+2,3)])

## Chiral Interaction Term: (S_i cross S_j) dot S_k

H_chiral = sps.csr_matrix((2**n,2**n))
for i in range(n-2):
    H_chiral += (op_dict[(i,2)]*op_dict[(i+1,3)]*op_dict[(i+2,1)] -
                 op_dict[(i,3)]*op_dict[(i+1,2)]*op_dict[(i+2,1)] +
                 op_dict[(i,3)]*op_dict[(i+1,1)]*op_dict[(i+2,2)] -
                 op_dict[(i,1)]*op_dict[(i+1,3)]*op_dict[(i+2,2)] +
                 op_dict[(i,1)]*op_dict[(i+1,2)]*op_dict[(i+2,3)] -
                 op_dict[(i,2)]*op_dict[(i+1,1)]*op_dict[(i+2,3)])#*(-1)**(i+1)

H_dict = {'H_chiral':H_chiral}

while J <= 1:
    H_chiral_full = sps.csr_matrix((2**n,2**n))
    H_chiral_full += J*H_dict['H_chiral']
    H = H_neighbor + H_chiral_full

    ## ## Solve for the Hamiltonian's eigenvalues/vectors and save them in lists
    
    eig = spslin.eigsh(H,k=10,which='SA')
    # print(eig[0],'\n')
    eigVals = np.sort(eig[0])
    eigVecs = eig[1]
    eigValList1.append(eigVals[0])
    eigList1.append(eigVals[1]-eigVals[0])
    eigValList2.append(eigVals[1])
    eigList2.append(eigVals[2]-eigVals[0])
    eigValList3.append(eigVals[3])
    eigList3.append(eigVals[3]-eigVals[0])
    eigValList4.append(eigVals[4])
    eigList4.append(eigVals[4]-eigVals[0])
    eigValList5.append(eigVals[5])
    eigList5.append(eigVals[5]-eigVals[0])
    JList.append(J)
    vecList.append(eigVecs[:,0])
    J += .02

endHamiltonian = time.time()
elapsed = endHamiltonian - start
print('This hamiltonian and its eigenvalues took',elapsed,'seconds to generate.',n,'spin system')

## ## Plot Energy Difference: E_1 - E_0

plt.Figure(figsize=(5,3))

# sns.set_style('ticks')
# sns.set_context('paper')

plt.xlabel('$J_\\chi$',fontsize='xx-large')
plt.ylabel('$E_n - E_0$',fontsize='xx-large')

E1, = plt.plot(JList,eigList1,label='$E_1 - E_0$',linewidth=4,color='k')
E3, = plt.plot(JList,eigList3,label='$E_3 - E_0$')
E2, = plt.plot(JList,eigList2,label='$E_2 - E_0$',color='y')
E4, = plt.plot(JList,eigList4,label='$E_4 - E_0$')
E5, = plt.plot(JList,eigList5,label='$E_5 - E_0$')
# plt.plot(JList,eigValList1,label='$E_0$')
# plt.plot(JList,np.gradient(eigValList1,JList),label='$E_0$ prime')
# plt.plot(JList,np.gradient(np.gradient(eigValList1,JList),JList),label='$E_0$ double prime')

handles = [E1,E2,E3,E4,E5]
labels = ['$E_1 - E_0$','$E_2 - E_0$','$E_3 - E_0$','$E_4 - E_0$','$E_5 - E_0$']
plt.tight_layout()
plt.legend(handles,labels)
plt.show()

def corr(j,k,vec):
    '''
    Determines the correlation between two particles
    by calculating expecval(S_j dot S_k) minus 
    expecval(S_j) dot expecval(S_k).
    Args:
        j: (integer) first particle 
        k: (integer) second particle
        vec: (np array) vector to calculate expectation 
            value with
    Returns:
        (float) value of the correlation function
    '''
    return ((np.conj(vec).dot((S_x(j)*S_x(k) + S_y(j)*S_y(k) + S_z(j)*S_z(k)).dot(vec))
        - np.conj(vec).dot((S_x(j) + S_y(j) + S_z(j)).dot(vec))*np.conj(vec).dot((S_x(k) 
        + S_y(k) + S_z(k)).dot(vec)))).real
