import numpy as np
import numpy.linalg as nplin
import scipy.sparse as sps
import scipy.sparse.linalg as spslin
import matplotlib.pyplot as plt
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

## ## Form the Hamiltonian

## Periodic Boundary Conditions

# H = np.zeros((2**n,2**n))
# for i in range(n-1):
	# H += 0.5*np.matmul(S_plus(i),S_minus(i+1)) + 0.5*np.matmul(S_minus(i),S_plus(i+1)) + np.matmul(S_z(i),S_z(i+1))
# H += 0.5*np.matmul(S_plus(n-1),S_minus(0)) + 0.5*np.matmul(S_minus(n-1),S_plus(0)) + np.matmul(S_z(n-1),S_z(0))

## Transverse Ising Model

eigList = []
vecList = []
bList = []
b = -3

H_a = -S_z(n-1)*S_z(0)
for i in range(n-1):
    H_a += -S_z(i)*S_z(i+1)
H_b = sps.csr_matrix((2**n,2**n))
while b <= 3:
    H_b = -(b/2)*S_x(n-1)
    for i in range(n-1):
        H_b += -(b/2)*S_x(i)
    H = H_a + H_b
   
   ## ## Solve the Hamiltonian's eigenvectors and corresponding eigenvalues

    eig = spslin.eigsh(H,k=2,which='SA')
    eigList.append(eig[0][1]-eig[0][0])
    bList.append(np.around(b,2))
    vecList.append(eig[1][:,0])
    b += 0.1
print(len(bList))
end = time.time()
elapsed = end - start
print('\n')
print('This Hamiltonian took',elapsed,'seconds to generate.',n,'spin system')
# print('The eigenvectors are (by row): ')
# for i in range(2):
	# print(eig[1][:,i])

## ## Plot Energy Gap

# plt.figure(figsize=(5,3))

# sns.set_style('ticks')
# sns.set_context('paper')

# plt.xlabel('$b$')
# plt.ylabel('$E_1 - E_0$')

# plt.plot(bList,eigList)
# plt.tight_layout()
# plt.show()

## ## Determine Wavefunctions

## Wavefunction as b --> negative infinity

# psi_x_down = np.array([1,-1])
# psi_inf_neg = np.array([1,-1])
# for i in range(n-1):
    # psi_inf_neg = np.kron(psi_inf_neg,psi_x_down)
# psi_inf_neg = (1/(2**(1/2))**n)*psi_inf_neg

## Wavefunction as b --> positive infinity

# psi_x_up = np.array([1,1])
# psi_inf_pos = np.array([1,1])
# for i in range(n-1):
    # psi_inf_pos = np.kron(psi_inf_pos,psi_x_up)
# psi_inf_pos = (1/(2**(1/2))**n)*psi_inf_pos

## Wavefunction with all z spin up

# z_up = np.array([1,0])
# psi_0_up = np.array([1,0])
# for i in range(n-1):
    # psi_0_up = np.kron(psi_0_up,z_up)

## Wavefunction with all z spin down

# z_down = np.array([0,1])
# psi_0_down = np.array([0,1])
# for i in range(n-1):
    # psi_0_down = np.kron(psi_0_down,z_down)

## ## Determine ground state overlap with asymptotic wavefunctions

## Overlap with b --> negative infinity wavefunction

# overlap_inf_neg = []
# for i in range(len(bList)):
    # overlap_inf_neg.append((np.conj(vecList[i].dot(psi_inf_neg))*vecList[i].dot(psi_inf_neg)))

## Overlap with b --> positive infinity wavefunction

# overlap_inf_pos = []
# for i in range(len(bList)):
    # overlap_inf_pos.append((np.conj(vecList[i].dot(psi_inf_pos))*vecList[i].dot(psi_inf_pos)))

## Overlap with up minus down

# overlap_minus = []
# for i in range(len(bList)):
    # overlap_minus.append((0.5*np.conj(vecList[i].dot(psi_0_up - psi_0_down))*vecList[i].dot((psi_0_up - psi_0_down))))

## Overlap with up plus down

# overlap_plus = []
# for i in range(len(bList)):
    # overlap_plus.append((0.5*np.conj(vecList[i].dot((psi_0_up + psi_0_down)))*vecList[i].dot((psi_0_up + psi_0_down))))

## Overlap with plus plus minus 

# overlap_plusminus = np.array(overlap_plus) + np.array(overlap_minus)

## ## Plot overlaps

# plt.xlabel('$b$')
# plt.ylabel('Overlap')

# plt.plot(bList,overlap_inf_pos,label='pos')
# plt.plot(bList,overlap_inf_neg,label='neg')
# plt.plot(bList,overlap_minus,label='minus')
# plt.plot(bList,overlap_plus,label='plus')
# plt.plot(bList,overlap_plusminus,label='plusminus',linestyle='--')

# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

## ## Find total spin in respective directions

# S_xt = sps.csr_matrix((2**n,2**n))
# for i in range(n):
    # S_xt += S_x(i)

# S_yt = sps.csr_matrix((2**n,2**n))
# for i in range(n):
    # S_yt += S_y(i)

# S_zt = sps.csr_matrix((2**n,2**n))
# for i in range(n):
    # S_zt += S_z(i)

## ## Find magnetization expectation values

# expecval_mx_list = []
# for i in range(len(bList)):
    # expecval_mx_list.append(vecList[i].dot(S_xt.dot(vecList[i])))

# expecval_my_list = []
# for i in range(len(bList)):
    # expecval_my_list.append(vecList[i].dot(S_yt.dot(vecList[i])))

# expecval_mz_list = []
# for i in range(len(bList)):
    # expecval_mz_list.append(vecList[i].dot(S_zt.dot(vecList[i])))

## ## Plot magnetization

# plt.xlabel('$b$')
# plt.ylabel('$m_x$ and $m_z$')

# plt.plot(bList,expecval_mx_list,label='$m_x$')
# plt.plot(bList,expecval_mz_list,label='$m_z$')

# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

## ## Create spin-spin correlation function

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

# corr_list1 = []
# for i in range(len(bList)):
    # corr_list1.append(corr(0,1,vecList[i]))

# corr_list2 = []
# for i in range(len(bList)):
    # corr_list2.append(corr(0,2,vecList[i]))

# corr_list3 = []
# for i in range(len(bList)):
    # corr_list3.append(corr(0,3,vecList[i]))

# corr_list4 = []
# for i in range(len(bList)):
    # corr_list4.append(corr(0,4,vecList[i]))
	
# corr_list5 = []
# for i in range(len(bList)):
    # corr_list5.append(corr(0,5,vecList[i]))

# corr_list6 = []
# for i in range(len(bList)):
    # corr_list6.append(corr(0,6,vecList[i]))

# corr_list7 = []
# for i in range(len(bList)):
    # corr_list7.append(corr(0,7,vecList[i]))

# corr_list8 = []
# for i in range(len(bList)):
    # corr_list8.append(corr(0,8,vecList[i]))

# corr_list9 = []
# for i in range(len(bList)):
    # corr_list9.append(corr(0,9,vecList[i]))

# corr_list10 = []
# for i in range(len(bList)):
    # corr_list10.append(corr(0,10,vecList[i]))

# plt.plot(bList,corr_list1)
# plt.plot(bList,corr_list2)
# plt.plot(bList,corr_list3)
# plt.plot(bList,corr_list4)
# plt.plot(bList,corr_list5)
# plt.plot(bList,corr_list6)
# plt.plot(bList,corr_list7)
# plt.plot(bList,corr_list8)
# plt.plot(bList,corr_list9)
# plt.plot(bList,corr_list10)
# plt.show()

## ## Determine the reduced density matrix

## Form total density matrix

# rhoList = []
# rhoeigList = []

# for i in range(len(bList)):
#     rhoList.append(np.outer(vecList[i],vecList[i]))

# for i in range(len(bList)):
#     rhoList[i] = rhoList[i].reshape([2,int((2**n)/2),2,int((2**n)/2)])
#     rhoList[i] = np.trace(rhoList[i],axis1=0,axis2=2)
#     rhoeigList.append(nplin.eig(rhoList[i])[0].real)

## ## Find the entanglement entropy

# EE = []

# for i in range(len(bList)):
#     EE.append(-rhoeigList[i][0]*np.log2(rhoeigList[i][0]))

# finish = time.clock()
# print('This code took',finish-start,'seconds to run.',n,'spin system.')

# plt.plot(bList,EE)
# plt.show()


# def EntEnt(vec,n_A=1):
#     '''
#     Calculates the Von Neumann entropy
#     of entanglement for a pure system
#     defined by vec
#     Args:
#         vec (list, np array, sparse array):
#             state vector for the system
#         which_sub_sys (string): which subsystem
#             we want to find entanglement entropy for.
#             Options are A and B; default is A
#         n_A (int): number of particles in
#             subsystem A; default is 1
#     Returns:
#         (float) entanglement entropy of
#             subsystem A in vec state
#     '''
#     n_B = n - n_A
#     d_A = 2**n_A
#     d_B = 2**n_B
    
#     rho = np.outer(vec,vec)
#     rho_tensor = rho.reshape([d_A,d_B,d_A,d_B])
#     red_rho = np.trace(rho_tensor,axis1=1,axis2=3)

#     eigVals = nplin.eigh(red_rho)[0]
#     entent = 0
#     for i in range(len(eigVals)):
#         entent += -eigVals[i]*np.log(eigVals[i])
#     print(entent)
#     return entent

# def EntEnt(vec,which_sub_sys='A',n_A=1):
#     '''
#     Calculates the Von Neumann entropy
#     of entanglement for a pure system
#     defined by vec
#     Args:
#         vec (list, np array, sparse array):
#             state vector for the system
#         which_sub_sys (string): which subsystem
#             we want to find entanglement entropy for.
#             Options are 'A' and 'B'; default is 'A'
#         n_A (int): number of particles in
#             subsystem A. Default is 1
#     Returns:
#         (float) entanglement entropy of
#             subsystem A in vec state
#     '''
#     n_B = n - n_A

#     if which_sub_sys == 'A':
#         d_A = 2 ** n_A
#         d_B = 2 ** n_B
#         d_C = 1
#     elif which_sub_sys == 'B':
#         d_A = 2 ** n_A
#         d_B = 2 ** n_B
#         d_C = 1

#     P = vec.reshape((d_A,d_B,d_C))
#     P1 = np.transpose(P,(0,2,1))
#     P2 = np.transpose(P,(1,0,2))
#     Q = np.dot(P1, np.conj(P))
#     if which_sub_sys == 'A':
#         rhoA = Q.flatten().reshape((d_A,d_A))
#     elif which_sub_sys == 'B':
#         rhoA = Q.flatten().reshape((d_B,d_B))
#     EE = 0
#     rho_eigList = nplin.eig(rhoA)[0]
    
#     for i in range(len(rho_eigList)):
#         if rho_eigList[i] == 0:
#             EE += 0
#         else:
#             EE += -rho_eigList[i]*np.log(rho_eigList[i])
#     return EE.real

# def EntEnt(vec,which_sub_sys='A',n_A=1):
#     '''
#     Calculates the Von Neumann entropy
#     of entanglement for a pure system
#     defined by vec
#     Args:
#         vec (list, np array, sparse array):
#             state vector for the system
#         which_sub_sys (string): which subsystem
#             we want to find entanglement entropy for.
#             Options are 'A' and 'B'; default is 'A'
#         n_A (int): number of particles in
#             subsystem A; default is 1
#     Returns:
#         (float) entanglement entropy of
#             subsystem A in vec state
#     '''
#     n_B = n - n_A

#     if which_sub_sys == 'A':
#         d_A = 2 ** n_A
#         d_B = 2 ** n_B

#     P = vec.reshape((d_A,d_B))
#     P1 = np.transpose(P,(0,1))
#     P2 = np.transpose(P,(1,0))
#     Q = np.dot(P1, np.conj(P2))
#     if which_sub_sys == 'A':
#         rhoA = Q.reshape((d_A,d_A))
#     elif which_sub_sys == 'B':
#         rhoA = Q.reshape((d_B,d_B))
#     EE = 0
#     rho_eigList = nplin.eig(rhoA)[0]
    
#     for i in range(len(rho_eigList)):
#         if rho_eigList[i] == 0:
#             EE += 0
#         else:
#             EE += -rho_eigList[i]*np.log(rho_eigList[i])
#     return EE.real

def EntEnt(vec,n_A=1):
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
    rho_reduced = np.zeros((d_A,d_A),dtype=complex)
    for i in range(len(basis_full)):
        rho_reduced += np.conj(basis_full[i]).dot(rho).dot(np.transpose(basis_full[i]))
    
    eigVals = nplin.eig(rho_reduced)[0]
    entent = 0
    for i in range(len(eigVals)):
        if eigVals[i] == 0:
            entent += 0
        else:
            entent += -eigVals[i]*np.log(eigVals[i])
    return entent.real
    
# EE_list = []
# for i in range(len(bList)):
#     EE_list.append(EntEnt(vecList[i],n_A=7))

# plt.plot(bList,EE_list)

EE_list = []
for i in range(len(bList)):
    EE_list.append(EntEnt(vecList[i],n_A=1))
plt.plot(bList,EE_list)
plt.title('EE vs b')
plt.show()

print('n_A = 1',EntEnt(vecList[bList.index(.5)],n_A=1))
print('n_A = 2',EntEnt(vecList[bList.index(.5)],n_A=2))
print('n_A = 3',EntEnt(vecList[bList.index(.5)],n_A=3))
print('n_A = 4',EntEnt(vecList[bList.index(.5)],n_A=4))
print('n_A = 5',EntEnt(vecList[bList.index(.5)],n_A=5))
print('n_A = 6',EntEnt(vecList[bList.index(.5)],n_A=6))
print('n_A = 7',EntEnt(vecList[bList.index(.5)],n_A=7))
# print('n_A = 8',EntEnt(vecList[bList.index(.5)],n_A=8))
# print('n_A = 9',EntEnt(vecList[bList.index(.5)],n_A=9))
# print('n_A = 10',EntEnt(vecList[bList.index(.5)],n_A=10))
# print('n_A = 11',EntEnt(vecList[bList.index(.5)],n_A=11))

EE_list2 = []

for i in range(n-1):
    EE_list2.append(EntEnt(vecList[bList.index(.5)],n_A=(i+1)))

plt.plot([x for x in range(1,n)],EE_list2)
plt.scatter([x for x in range(1,n)],EE_list2)
plt.Figure(figsize=(5,3))

plt.xlabel('Subsystem $A$ size',fontsize='xx-large')
plt.ylabel('$S_{AB}$',fontsize='xx-large')
plt.title('Entanglement Entropy vs. Subsystem Size; $b = 0.5$')
plt.tight_layout()
plt.show()

## ## Find overlap (fidelity) between Psi(J) and Psi(J + delta(J))

# overlapList = []

# for i in range(len(bList)-1):
#     x = np.inner(np.conj(vecList[i]),np.transpose(vecList[i+1]))
#     overlapList.append((np.conj(x)*x).real)

# bList_new = copy.deepcopy(bList)
# bList_new.pop(-1)

# # overlapPrime = np.gradient(overlapList,bList_new)

# plt.plot(bList_new,overlapList,label='original')
# # plt.plot(JList_new,overlapPrime,label='derivative')
# plt.Figure(figsize=(5,3))

# sns.set_style('ticks')
# sns.set_context('paper')

# plt.xlabel('$b$')
# plt.ylabel('Overlap')
# plt.tight_layout()
# plt.legend()
# # plt.axis((0,1,0,1.2))

# plt.show()
