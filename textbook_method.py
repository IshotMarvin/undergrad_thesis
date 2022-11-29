import numpy as np
import time

n = int(input("Enter number of spins: "))

start = time.clock()

## Form the basis

basis = np.identity(2**n)

## Find the matrix representation of the spin operators for each particle

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
		(numpy array) matrix representation
			of the S_plus operator acting
			on the given particle
	'''
	S_plus = np.eye(2,k=1)
	for i in range(m):
		S_plus = np.kron(np.identity(2),S_plus)
	for i in range(n-m-1):
		S_plus = np.kron(S_plus,np.identity(2))
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
		(numpy array) matrix representation
			of the S_plus operator acting
			on the given particle
	'''
	S_minus = np.eye(2,k=-1)
	for i in range(m):
		S_minus = np.kron(np.identity(2),S_minus)
	for i in range(n-m-1):
		S_minus = np.kron(S_minus,np.identity(2))
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
		(numpy array) matrix representation 
		of the S_z operator acting on the
		given particle
	'''
	S_z = np.zeros([2,2])
	S_z[0,0] = 1
	S_z[1,1] = -1
	for i in range(m):
		S_z = np.kron(np.identity(2),S_z)
	for i in range(n-m-1):
		S_z = np.kron(S_z,np.identity(2))
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
        (numpy array) matrix representation 
        of the S_x operator acting on the
        given particle
    '''
    S_x = np.zeros([2,2])
    S_x[0,1] = 1
    S_x[1,0] = 1
    for i in range(m):
        S_x = np.kron(np.identity(2),S_x)
    for i in range(n-m-1):
        S_x = np.kron(S_x,np.identity(2))
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
		(numpy array) matrix representation 
		of the S_y operator acting on the
		given particle
	'''
	return (S_plus(m) - S_minus(m))/2j

## Find the matrix representation of the total spin operators

# S_plus_total = np.zeros((2**n,2**n))
# for i in range(n):
	# S_plus_total += S_plus(i)
	
# S_minus_total = np.transpose(S_plus_total)

# S_z_total = np.zeros((2**n,2**n))
# for i in range(n):
	# S_z_total += S_z(i)

# S_x_total = np.zeros((2**n,2**n))
# for i in range(n):
	# S_x_total += S_z(i)
	
# S_y_total = np.zeros((2**n,2**n))
# for i in range(n):
	# S_y_total += S_z(i)
	
## Form the Hamiltonian

## Periodic Boundary Conditions

# H = np.zeros((2**n,2**n))
# for i in range(n-1):
	# H += 0.5*np.matmul(S_plus(i),S_minus(i+1)) + 0.5*np.matmul(S_minus(i),S_plus(i+1)) + np.matmul(S_z(i),S_z(i+1))
# H += 0.5*np.matmul(S_plus(n-1),S_minus(0)) + 0.5*np.matmul(S_minus(n-1),S_plus(0)) + np.matmul(S_z(n-1),S_z(0))

## Transverse Ising Model

b = 1
H = np.zeros((2**n,2**n))
for i in range(n):
    H += -(b/2)*S_x(i)
for i in range(n-1):
    H += -np.matmul(S_z(i),S_z(i+1))
H += -np.matmul(S_z(n-1),S_z(0))

print('The Hamiltonian matrix is: ')
print(H)

## Solve the Hamiltonian's eigenvectors and corresponding eigenvalues

eig = np.linalg.eig(H)
print('\n')
# print('The eigenvectors are (by row): ')
# for i in range(2**n):
	# print(eig[1][:,i])
print('\n')
print('The corresponding energy eigenvalues are: ')
print(eig[0])

end = time.clock()
elapsed = end - start
print('\n')
print('This code took',elapsed,'seconds to run.',n,'spin system')
