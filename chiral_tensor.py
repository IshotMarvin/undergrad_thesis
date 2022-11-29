import numpy as np
import numpy.linalg as nplin
import scipy.sparse as sps
import scipy.sparse.linalg as spslin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax
# import seaborn as sns
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

## Nearest Neighbor (Heisenberg) coupling: S dot S_neighbors; periodic boundary conditions

H_neighbor = (op_dict[(n-2,1)]*op_dict[(n-1,1)] +
              op_dict[(n-2,2)]*op_dict[(n-1,2)] +
              op_dict[(n-2,3)]*op_dict[(n-1,3)] +
              op_dict[(0,1)]*op_dict[(n-2,1)] +
              op_dict[(0,2)]*op_dict[(n-2,2)] +
              op_dict[(0,3)]*op_dict[(n-2,3)] +
              op_dict[(0,1)]*op_dict[(n-1,1)] +
              op_dict[(0,2)]*op_dict[(n-1,2)] +
              op_dict[(0,3)]*op_dict[(n-1,3)] +
              op_dict[(1,1)]*op_dict[(n-1,1)] +
              op_dict[(1,2)]*op_dict[(n-1,2)] +
              op_dict[(1,3)]*op_dict[(n-1,3)])

for i in range(n-2):
    H_neighbor += (op_dict[(i,1)]*op_dict[(i+1,1)] +
                   op_dict[(i,2)]*op_dict[(i+1,2)] +
                   op_dict[(i,3)]*op_dict[(i+1,3)] +
                   op_dict[(i,1)]*op_dict[(i+2,1)] +
                   op_dict[(i,2)]*op_dict[(i+2,2)] +
                   op_dict[(i,3)]*op_dict[(i+2,3)])

## Chiral Interaction Term: (S_i cross S_j) dot S_k

H_chiral = (op_dict[(n-2,2)]*op_dict[(n-1,3)]*op_dict[(0,1)] -
            op_dict[(n-2,3)]*op_dict[(n-1,2)]*op_dict[(0,1)] +
            op_dict[(n-2,3)]*op_dict[(n-1,1)]*op_dict[(0,2)] -
            op_dict[(n-2,1)]*op_dict[(n-1,3)]*op_dict[(0,2)] +
            op_dict[(n-2,1)]*op_dict[(n-1,2)]*op_dict[(0,3)] -
            op_dict[(n-2,2)]*op_dict[(n-1,1)]*op_dict[(0,3)] +
            op_dict[(n-1,2)]*op_dict[(0,3)]*op_dict[(1,1)] -
            op_dict[(n-1,3)]*op_dict[(0,2)]*op_dict[(1,1)] +
            op_dict[(n-1,3)]*op_dict[(0,1)]*op_dict[(1,2)] -
            op_dict[(n-1,1)]*op_dict[(0,3)]*op_dict[(1,2)] +
            op_dict[(n-1,1)]*op_dict[(0,2)]*op_dict[(1,3)] -
            op_dict[(n-1,2)]*op_dict[(0,1)]*op_dict[(1,3)])

for i in range(n-2):
    H_chiral += (op_dict[(i,2)]*op_dict[(i+1,3)]*op_dict[(i+2,1)] -
                 op_dict[(i,3)]*op_dict[(i+1,2)]*op_dict[(i+2,1)] +
                 op_dict[(i,3)]*op_dict[(i+1,1)]*op_dict[(i+2,2)] -
                 op_dict[(i,1)]*op_dict[(i+1,3)]*op_dict[(i+2,2)] +
                 op_dict[(i,1)]*op_dict[(i+1,2)]*op_dict[(i+2,3)] -
                 op_dict[(i,2)]*op_dict[(i+1,1)]*op_dict[(i+2,3)])#*(-1)**(i+1)

H_dict = {'H_chiral':H_chiral}

while J <= 1.02:
    H_chiral_full = sps.csr_matrix((2**n,2**n))
    H_chiral_full += J*H_dict['H_chiral']
    H = H_neighbor + H_chiral_full

    ## ## Solve for the Hamiltonian's eigenvalues/vectors and save them in lists
    
    eig = spslin.eigsh(H,k=20,which='SA')
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
    JList.append(np.around(J,2))
    vecList.append(eigVecs[:,0])
    J += .02

del H_dict

endHamiltonian = time.time()
elapsed = endHamiltonian - start
print('This hamiltonian and its eigenvalues took',elapsed,'seconds to generate.',n,'spin system')

## ## Plot Energy Difference: E_n - E_0

plt.Figure(figsize=(5,3))

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

## ## Find spin-spin correlation

## Create spin-spin correlation function

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
    return ((np.conj(vec).dot((op_dict[(j,1)]*op_dict[(k,1)] + op_dict[(j,2)]*op_dict[(k,2)]
                               + op_dict[(j,3)]*op_dict[(k,3)]).dot(vec)) -
             np.conj(vec).dot((op_dict[(j,1)] + op_dict[(j,2)] +
                        op_dict[(j,3)]).dot(vec))*np.conj(vec).dot((op_dict[(k,1)] +
                                                                    op_dict[(k,2)] +
                                                                    op_dict[(k,3)]).dot(vec)))).real

def corr_fourier(particle,k_y,k_x,vec):
    '''
    Determines the fourier transform of the correlaion
    function. 
    Args:
        particle: (int) the particle that we are 
                        calculating the fourier
                        correlation function for
        k_y: (float) the y-component of the k vector;
                     values are 0 and pi
        k_x: (float) the x-component of the k vector;
                     values should range from 0 to 2pi
        vec: (np array) vector to calculate expectation
                        value with
    Returns:
        (float) value of the fourier transform of the
                correlation function
    '''
    N_x = n/2
    S_k = 0
    for i in range(n):
        if i == particle:
            pass
        else:
            S_k += corr(particle,i,vec)*np.exp(-k_x*i*1j)*np.exp(k_y*1j)
    return (1/n)*S_k.real

## Evaluate the correlation function between the first particle and the rest

N_x = n/2

k_x_list = np.linspace(0,2*np.pi,N_x)

corr_four1 = []
for i in range(len(k_x_list)):
    corr_four1.append(corr_fourier(0,0,k_x_list[i],vecList[JList.index(0)]))

corr_four2 = []
for i in range(len(k_x_list)):
    corr_four2.append(corr_fourier(0,np.pi,k_x_list[i],vecList[JList.index(0)]))

plt.plot(k_x_list,corr_four1,label='$k_y = 0$')
plt.plot(k_x_list,corr_four2,label='$k_y = \pi$',marker=10)
plt.scatter(k_x_list,corr_four1,marker='^',s=150)
plt.scatter(k_x_list,corr_four2,marker='o',s=150)

plt.xlabel('$k_x$',fontsize='xx-large')
plt.ylabel('Fourier Correlation $1,n$',fontsize='xx-large')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

corr_list1 = []
for i in range(len(JList)):
    corr_list1.append(corr(0,1,vecList[i]))
# deriv1 = np.gradient(corr_list1,JList)

corr_list2 = []
for i in range(len(JList)):
    corr_list2.append(corr(0,2,vecList[i]))
# deriv2 = np.gradient(corr_list2,JList)
    
corr_list3 = []
for i in range(len(JList)):
    corr_list3.append(corr(0,3,vecList[i]))
# deriv3 = np.gradient(corr_list3,JList)
    
corr_list4 = []
for i in range(len(JList)):
    corr_list4.append(corr(0,4,vecList[i]))
# deriv4 = np.gradient(corr_list4,JList)
    
corr_list5 = []
for i in range(len(JList)):
    corr_list5.append(corr(0,5,vecList[i]))
# deriv5 = np.gradient(corr_list5,JList)

corr_list6 = []
for i in range(len(JList)):
    corr_list6.append(corr(0,6,vecList[i]))
# deriv6 = np.gradient(corr_list6,JList)

corr_list7 = []
for i in range(len(JList)):
    corr_list7.append(corr(0,7,vecList[i]))
# deriv7 = np.gradient(corr_list7,JList)
    
corr_list8 = []
for i in range(len(JList)):
    corr_list8.append(corr(0,8,vecList[i]))
# deriv8 = np.gradient(corr_list8,JList)

corr_list9 = []
for i in range(len(JList)):
    corr_list9.append(corr(0,9,vecList[i]))
# deriv9 = np.gradient(corr_list9,JList)

corr_list10 = []
for i in range(len(JList)):
    corr_list10.append(corr(0,10,vecList[i]))
# deriv10 = np.gradient(corr_list10,JList)
    
corr_list11 = []
for i in range(len(JList)):
    corr_list11.append(corr(0,11,vecList[i]))
# deriv11 = np.gradient(corr_list11,JList)

plt.plot(JList,corr_list1,label='n = 2')
# # plt.plot(JList,deriv1,label='n = 1 prime')
plt.plot(JList,corr_list2,label='n = 3')
# # plt.plot(JList,deriv2,label='n = 2 prime')
plt.plot(JList,corr_list3,label='n = 4')
# # plt.plot(JList,deriv3,label='n = 3 prime')
plt.plot(JList,corr_list4,label='n = 5')
# # plt.plot(JList,deriv4,label='n = 4 prime')
plt.plot(JList,corr_list5,label='n = 6')
# # plt.plot(JList,deriv5,label='n = 5 prime')
plt.plot(JList,corr_list6,label='n = 7',linestyle='-.')
# plt.plot(JList,deriv6,label='n = 7 prime', linestyle='-.')
plt.plot(JList,corr_list7,label='n = 8')
# plt.plot(JList,deriv7,label='n = 8 prime')
plt.plot(JList,corr_list8,label='n = 9',linestyle=':')
# # plt.plot(JList,deriv8,label='n = 9 prime')
plt.plot(JList,corr_list9,label='n = 10')
# # plt.plot(JList,deriv9,label='n = 10 prime')
plt.plot(JList,corr_list10,label='n = 11')
# # plt.plot(JList,deriv10,label='n = 11 prime')
plt.plot(JList,corr_list11,label='n = 12',linestyle='--')
# # plt.plot(JList,deriv11,label='n = 12 prime', linestyle='--')

plt.xlabel('$J_\\chi$',fontsize='xx-large')
plt.ylabel('Correlation $1,n$',fontsize='xx-large')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

## ## Determine the reduced density matrix; subsystem A is the fist particle,
## ## subsystem B is the rest

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

    if which_sub_sys == 'A':
        d_A = 2 ** n_A
        d_B = 2 ** n_B

    P = vec.reshape((d_A,d_B))
    P1 = np.transpose(P,(0,1))
    P2 = np.transpose(P,(1,0))
    Q = np.dot(P1, np.conj(P2))
    if which_sub_sys == 'A':
        rhoA = Q.reshape((d_A,d_A))
    elif which_sub_sys == 'B':
        rhoA = Q.reshape((d_B,d_B))
    EE = 0
    rho_eigList = nplin.eigvals(rhoA)
    
    for i in range(len(rho_eigList)):
        if rho_eigList[i] == 0:
            EE += 0
        else:
            EE += -rho_eigList[i]*np.log(rho_eigList[i])
    return EE.real

print('n_A = 1',EntEnt(vecList[JList.index(.5)],n_A=1))
print('n_A = 2',EntEnt(vecList[JList.index(.5)],n_A=2))
print('n_A = 3',EntEnt(vecList[JList.index(.5)],n_A=3))
print('n_A = 4',EntEnt(vecList[JList.index(.5)],n_A=4))
print('n_A = 5',EntEnt(vecList[JList.index(.5)],n_A=5))
print('n_A = 6',EntEnt(vecList[JList.index(.5)],n_A=6))
print('n_A = 7',EntEnt(vecList[JList.index(.5)],n_A=7))
print('n_A = 8',EntEnt(vecList[JList.index(.5)],n_A=8))
print('n_A = 9',EntEnt(vecList[JList.index(.5)],n_A=9))
print('n_A = 10',EntEnt(vecList[JList.index(.5)],n_A=10))
print('n_A = 11',EntEnt(vecList[JList.index(.5)],n_A=11))

EE_list2 = []

for i in range(n-1):
    EE_list2.append(EntEnt(vecList[JList.index(.5)],n_A=(i+1)))

plt.plot([x for x in range(1,n)],EE_list2)
plt.scatter([x for x in range(1,n)],EE_list2)
plt.Figure(figsize=(5,3))

plt.xlabel('Subsystem $A$ size',fontsize='xx-large')
plt.ylabel('$S_{AB}$',fontsize='xx-large')
plt.title('Entanglement Entropy vs. Subsystem Size; $J_\\chi = 0.5$')
plt.tight_layout()
plt.show()

EE_list = []
for i in range(len(JList)):
    # print(vecList[i])
    EE_list.append(EntEnt(vecList[i],n_A=1))

plt.plot(JList,EE_list)
plt.Figure(figsize=(5,3))

plt.xlabel('$J_\\chi$',fontsize='xx-large')
plt.ylabel('$S_{AB}$',fontsize='xx-large')
plt.tight_layout()
plt.show()

## ## Find overlap (fidelity) between Psi(J) and Psi(J + delta(J))

overlapList = []

for i in range(len(JList)-1):
    x = np.inner(np.conj(vecList[i]),np.transpose(vecList[i+1]))
    overlapList.append((np.conj(x)*x).real)

JList_new = copy.deepcopy(JList)
JList_new.pop(-1)

overlapPrime = np.gradient(overlapList,JList_new)

plt.plot(JList_new,overlapList)
# plt.plot(JList_new,overlapPrime,label='derivative')
plt.Figure(figsize=(5,3))

plt.xlabel('$J_\\chi$',fontsize='xx-large')
plt.ylabel('Wave function fidelity',fontsize='xx-large')
plt.tight_layout()
# plt.legend()
# plt.axis((0,1,0,1.2))

plt.show()
