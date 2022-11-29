import numpy as np
import numpy.linalg as nplin
import scipy.sparse as sps
import scipy.sparse.linalg as spslin
import matplotlib.pyplot as plt
import time

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
	for i in range(m):
       	    S_plus = sps.kron(sps.identity(2),S_plus)
	for i in range(n-m-1):
       	    S_plus = sps.kron(S_plus,sps.identity(2))
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
	for i in range(m):
	    S_minus = sps.kron(np.identity(2),S_minus)
	for i in range(n-m-1):
	    S_minus = sps.kron(S_minus,sps.identity(2))
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
    S_z = sps.kron(sps.identity(2**(m)),S_z)
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

## ## Form the Hamiltonian: Nearest Neighbor Coupling

eigList = []
eigGs = []
vecList = []
JList = []
J = 0

## Nearest Neighbor Coupling: S dot S_neighbors

# H = (op_dict[(0,1)]*op_dict[(n-1,1)] +
#      op_dict[(0,2)]*op_dict[(n-1,2)] +
#      op_dict[(0,3)]*op_dict[(n-1,3)])
H = sps.csr_matrix((2**n,2**n))             
for i in range(n-1):
    H += (op_dict[(i,1)]*op_dict[(i+1,1)] +
          op_dict[(i,2)]*op_dict[(i+1,2)] +
          op_dict[(i,3)]*op_dict[(i+1,3)])

# while J <= 1.1:
    
#     H = (S_x(n-2)*S_x(n-1) + S_y(n-2)*S_y(n-1) + S_z(n-2)*S_z(n-1))
#     for i in range(n-2):
#         H += (S_x(i)*S_x(i+1) + S_y(i)*S_y(i+1) + S_z(i)*S_z(i+1) +
#               S_x(i)*S_x(i+2) + S_y(i)*S_y(i+2) + S_z(i)*S_z(i+2))
#     print(H)

## ## Solve for the Hamiltonian's eigenvalues/vectors and save them in lists

H = H.todense()
eig = nplin.eigh(H)
eigList.append(eig[0][1] - eig[0][0])
eigGs.append(eig[0][0])
JList.append(J)
# print(eig[1][:,0])
vecList.append(eig[1][:,0])
print(eig[0])
J += .1

endHamiltonian = time.time()
elapsed = endHamiltonian - start
print('This hamiltonian and its eigenvalues took',elapsed,'seconds to generate.',n,'spin system')

## ## Plot Energy Difference: E_1 - E_0

# plt.figure(figsize=(5,3))

# plt.xlabel('$J$')
# plt.ylabel('$E_1 - E_0$')

# plt.plot(JList,eigList)
# plt.tight_layout()
# plt.show()

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
    return ((np.conj(vec).dot((S_x(j)*S_x(k) + S_y(j)*S_y(k) + S_z(j)*S_z(k)).dot(vec))
        - np.conj(vec).dot((S_x(j) + S_y(j) + S_z(j)).dot(vec))*np.conj(vec).dot((S_x(k) 
        + S_y(k) + S_z(k)).dot(vec)))).real

blah = 0
for i in range(n-1):
        blah += (corr(i,i+1,vecList[0]))
blah += corr(0,n-1,vecList[0])
print('sum of correlation',blah)

print('gs energy',eigGs[0])
print('correlation 1,2',corr(0,1,vecList[0]))
print('correlation 2,3',corr(1,2,vecList[0]))
print('correlation 1,12',corr(1,n-1,vecList[0]))

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
    N_x = n
    S_k = 0
    for i in range(n):
        if i == particle:
            pass
        else:
            S_k += corr(particle,i,vec)*np.exp(-k_x*(i)*1j)
    return (1/n)*S_k.real

## Evaluate the correlation function between the first particle and the rest

corr_dict = {}

N_x = n
# k_x_list = []
# for i in range(int(n)):
#     k_x_list.append(2*np.pi*(1/N_x)*(-N_x + i)/2)

# k_x_list = np.linspace(-np.pi,((n/2)-1)/2,6)

k_x_list = np.linspace(0,2*np.pi,N_x+1)

corr_four1 = []
for i in range(len(k_x_list)):
    corr_four1.append(corr_fourier(0,0,k_x_list[i],vecList[0]))

plt.plot(k_x_list,corr_four1,label='$k_y = 0$')
plt.scatter(k_x_list,corr_four1)

plt.xlabel('$k_x$',fontsize='xx-large')
plt.ylabel('Fourier Correlation $1,n$',fontsize='xx-large')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

corr_list = []
for i in range(n):
        if i == 0:
                pass
        else:
                corr_list.append(corr(0,i,vecList[0]))

plt.plot([x for x in range(2,n+1)],corr_list)
plt.scatter([x for x in range(2,n+1)],corr_list)
plt.xlabel('$n$')
plt.ylabel('Correlatin $1,n$')
plt.show()
## Evaluate the correlation function between the first particle and the rest

# corr_list1 = []
# for i in range(len(JList)):
#     corr_list1.append(corr(0,1,vecList[i]))

# corr_list2 = []
# for i in range(len(JList)):
#     corr_list2.append(corr(0,2,vecList[i]))

# corr_list3 = []
# for i in range(len(JList)):
#     corr_list3.append(corr(0,3,vecList[i]))

# corr_list4 = []
# for i in range(len(JList)):
#     corr_list4.append(corr(0,4,vecList[i]))
	
# corr_list5 = []
# for i in range(len(JList)):
#     corr_list5.append(corr(0,5,vecList[i]))

# corr_list6 = []
# for i in range(len(JList)):
#     corr_list6.append(corr(0,6,vecList[i]))

# corr_list7 = []
# for i in range(len(JList)):
#     corr_list7.append(corr(0,7,vecList[i]))

# corr_list8 = []
# for i in range(len(JList)):
#     corr_list8.append(corr(0,8,vecList[i]))

# corr_list9 = []
# for i in range(len(JList)):
#     corr_list9.append(corr(0,9,vecList[i]))

# corr_list10 = []
# for i in range(len(JList)):
#     corr_list10.append(corr(0,10,vecList[i]))

# corr_list11 = []
# for i in range(len(JList)):
#     corr_list11.append(corr(0,11,vecList[i]))
    
# plt.plot(JList,corr_list1,label='n = 2')
# plt.plot(JList,corr_list2,label='n = 3')
# plt.plot(JList,corr_list3,label='n = 4')
# plt.plot(JList,corr_list4,label='n = 5')
# plt.plot(JList,corr_list5,label='n = 6')
# plt.plot(JList,corr_list6,label='n = 7',linestyle='-.')
# plt.plot(JList,corr_list7,label='n = 8')
# plt.plot(JList,corr_list8,label='n = 9',linestyle=':')
# plt.plot(JList,corr_list9,label='n = 10')
# plt.plot(JList,corr_list10,label='n = 11')
# plt.plot(JList,corr_list11,label='n = 12',linestyle='--')

# plt.xlabel('$J$')
# plt.ylabel('Correlation $1,n$')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

## ## Determine the reduced density matrix; subsystem A is the fist particle,
## ## subsystem B is the rest

## Set up the basis for subsystem B

# identity_B = np.identity(2**(n-1))
# basis_B = []

# for i in range(2**(n-1)):
#         basis_B.append(sps.kron(sps.identity(2),identity_B[:][i]))

## Form the density matrix, then the reduced density matrix,
## find its eigenvalues, then calculate the Von Neumann entropy

# EE_list = []
# for j in range(len(JList)):
#         rho_eigList = []
#         EE = 0
#         rho_reduced = sps.csr_matrix(np.zeros((2,2)))

#         rho = sps.csr_matrix(np.transpose(sps.csr_matrix(vecList[j].real)) * sps.csr_matrix(vecList[j].real))

#         for i in range(len(basis_B)):
#                 rho_reduced += basis_B[i] * rho * np.transpose(basis_B[i])

#         rho_reduced = rho_reduced.toarray()
#         rho_eigList = nplin.eig(rho_reduced)[0]
#         print(rho_reduced)
#         for i in range(2):
#                 EE += -rho_eigList[i]*np.log2(rho_eigList[i])
#         EE_list.append(EE)
# print(EE_list)
# plt.figure(figsize=(5,3))

# plt.xlabel('$J$')
# plt.ylabel('EE')

# plt.plot(JList,EE_list)
# plt.tight_layout()
# plt.show()

# n_A = 1
# n_B = n - n_A

# d_A = 2 ** n_A
# d_B = 2 ** n_B
# d_C = 1
# P = vecList[0].reshape((d_A,d_B,d_C))

# P1 = np.transpose(P,(0,2,1))
# P2 = np.transpose(P,(1,0,2))

# Q = np.dot(P1,np.conj(P))

# rhoA = Q.flatten().reshape((2,2))

# print(rhoA)

# EE = 0
# rho_eigList = nplin.eig(rhoA)[0]

# print(rho_eigList)

# for i in range(len(rho_eigList)):
#     EE += -rho_eigList[i]*np.log(rho_eigList[i])
# print(EE.real)

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
            subsystem A. Default is 1
    Returns:
        (float) entanglement entropy of
            subsystem A in vec state
    '''
    n_B = n - n_A

    if which_sub_sys == 'A':
        d_A = 2 ** n_A
        d_B = 2 ** n_B
        d_C = 1
    elif which_sub_sys == 'B':
        d_A = 1
        d_B = 2 ** n_A
        d_C = 2 ** n_B

    P = vec.reshape((d_A,d_B,d_C))
    P1 = np.transpose(P,(0,2,1))
    P2 = np.transpose(P,(1,0,2))
    Q = np.dot(P1, np.conj(P))
    rhoA = Q.flatten().reshape((2,2))

    EE = 0
    rho_eigList = nplin.eig(rhoA)[0]
    
    for i in range(len(rho_eigList)):
        if rho_eigList[i] == 0:
                EE += 0
        else:
                EE += -rho_eigList[i]*np.log(rho_eigList[i])
    return EE.real

EE_list = []
for i in range(len(JList)):
    EE_list.append(EntEnt(vecList[i]))

plt.plot(JList,EE_list)
plt.Figure(figsize=(5,3))

# plt.show()
