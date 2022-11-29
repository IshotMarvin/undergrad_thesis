import numpy as np
import numpy.linalg as nplin
import scipy.sparse as sps
import scipy.sparse.linalg as spslin
import matplotlib.pyplot as plt
import seaborn as sns
import time

n = int(input("Enter number of spins: "))

start = time.time()

## ## Form the basis

basis = sps.identity(2**n)

## ## Find the matrix representation of the spin operators for each particle

def S_plus(m,N=n):
	'''
	Produces the matrix representation of 
	the S_plus operator, which raises a 
	spin down state to up, or annihilates
	the state if it's already spin up.
	Args:
	    m: (integer) which particle the 
		operator is acting on
            N: (integer) number of particles
                in the system; default is
                the user input, n
	Returns:
            (sparse array) matrix representation
            of the S_plus operator acting
	    on the given particle
	'''
	S_plus = sps.eye(2,k=1)
	for i in range(m):
       	    S_plus = sps.kron(sps.identity(2),S_plus)
	for i in range(N-m-1):
       	    S_plus = sps.kron(S_plus,sps.identity(2))
	return S_plus

def S_minus(m,N=n):
	'''
	Produces the matrix representation of 
	the S_minus operator, which lowers a 
	spin up state to down, or annihilates
	the state if it's already spin down.
	Args:
	    m: (integer) which particle the 
		operator is acting on
            N: (integer) number of particles
                in the system; default is
                the user input, n
	Returns:
	    (sparse array) matrix representation
	    of the S_plus operator acting
       	    on the given particle
	'''
	S_minus = sps.eye(2,k=-1)
	for i in range(m):
	    S_minus = sps.kron(np.identity(2),S_minus)
	for i in range(N-m-1):
	    S_minus = sps.kron(S_minus,sps.identity(2))
	return S_minus

def S_z(m,N=n):
    '''
    Produces the matrix representation of
    the S_z operator, which finds the value
    of spin for the given particle in the z
    direction.
    Args:
        m: (integer) which particle the 
            operator is acting on
        N: (integer) number of particles
            in the system; default is
            the user input, n
    Returns:
        (sparse array) matrix representation 
        of the S_z operator acting on the
        given particle
    '''
    S_z = sps.diags([1,-1])
    S_z = sps.kron(sps.identity(2**(m)),S_z)
    S_z = sps.kron(S_z,sps.identity(2**(N-m-1)))
    return 1/2*S_z

def S_x(m,N=n):
    '''
    Produces the matrix representation of
    the S_x operator, which finds the value
    of spin for the given particle in the x
    direction.
    Args:
        m: (integer) which particle the 
            operator is acting on
        N: (integer) number of particles
            in the system; default is
            the user input, n
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
    S_x = sps.kron(S_x,sps.identity(2**(N-m-1)))
    return 1/2*S_x

def S_y(m,N=n):
    '''
    Produces the matrix representation of
    the S_y operator, which finds the value
    of spin for the given particle in the y
    direction.
    Args:
        m: (integer) which particle the 
            operator is acting on
        N: (integer) number of particles
            in the system; default is
            the user input, n
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
    S_y = sps.kron(S_y,sps.identity(2**(N-m-1)))
    return 1/2*S_y

## ## Form the Hamiltonian: Chiral Interaction with Nearest Neighbor Coupling

# eigList = []
# vecList = []
# JList = []

for u in range(5):
        eigList = []
        vecList = []
        JList = []
        J = 0
        n_new = n + 2*u
        
        ## Nearest Neighbor Coupling: S dot S_neighbors

        H_neighbor = 0
        H_neighbor = (S_x(n_new-2,n_new)*S_x(n_new-1,n_new) + S_y(n_new-2,n_new)*S_y(n_new-1,n_new) +
                     S_z(n_new-2,n_new)*S_z(n_new-1,n_new))
        for i in range(n_new-2):
                H_neighbor += (S_x(i,n_new)*S_x(i+1,n_new) + S_y(i,n_new)*S_y(i+1,n_new) +
                               S_z(i,n_new)*S_z(i+1,n_new) + S_x(i,n_new)*S_x(i+2,n_new) +
                               S_y(i,n_new)*S_y(i+2,n_new) + S_z(i,n_new)*S_z(i+2,n_new))

        ## Chiral Interaction Term: (S_i cross S_j) dot S_k
        H_chiral = 0
        while J <= 1:
                H_chiral = sps.csr_matrix((2**n_new,2**n_new))
                for i in range(n_new-2):
                        H_chiral += J*(S_y(i,n_new)*S_z(i+1,n_new)*S_x(i+2,n_new) -
                                       S_z(i,n_new)*S_y(i+1,n_new)*S_x(i+2,n_new) +
                                       S_z(i,n_new)*S_x(i+1,n_new)*S_y(i+2,n_new) -
                                       S_x(i,n_new)*S_z(i+1,n_new)*S_y(i+2,n_new) +
                                       S_x(i,n_new)*S_y(i+1,n_new)*S_z(i+2,n_new) -
                                       S_y(i,n_new)*S_x(i+1,n_new)*S_z(i+2,n_new))#*(-1)**(i+1)
                H = H_neighbor + H_chiral

                ## ## Solve for the Hamiltonian's eigenvalues/vectors and save them in lists
                                
                eig = spslin.eigsh(H,k=6,which='SA')
                # print(eig[0],'\n')
                eigVals = np.sort(eig[0])
                eigList.append(eigVals[5]-eigVals[0])
                JList.append(J)
                for i in range(1):
                        vecList.append(eig[1][:,i])
                J += .05
                
        ## ## Plot Energy Difference: E_5 - E_0

        plt.Figure(figsize=(5,3))
        sns.set_style('ticks')
        sns.set_context('paper')
                
        plt.xlabel('$J$')
        plt.ylabel('$E_5 - E_0$')

        plt.plot(JList,eigList,label=('n = %.0f' %n_new))

endHamiltonian = time.time()
elapsed = endHamiltonian - start
print('This hamiltonian and its eigenvalues took',elapsed,'seconds to generate and plot.',n,'spin system')

plt.legend(loc='best')
plt.tight_layout()
plt.show()

