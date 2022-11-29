import numpy as np
import time
import copy


## User inputs number of spins

n = int(input("Enter number of spins: "))

start = time.clock()

## Determine basis

basis = []
for i in range(2**n):   
	j = "{0:b}".format(i)         #convert index to binary (string)
	j = list(j)    				  #convert string to list
	while len(j) < n:             #ensure list is proper dimension (number of spins)
		j = [0] + j
	for k in range(len(j)):       #invert the binary values (making spin up 1 and spin down 0)
		if j[k] == "1":
			j[k] = 0
		else:	
			j[k] = 1
	basis.append(j)

basis = np.asarray(basis)

print('The basis for this spin system is: ')
print(basis)
print()
## Define the action of spin operators on the basis states

def vec(n):
	'''
	Picks out a vector which is an element of the basis.
	Args:
		n: (integer) which vector to pick
	Returns:
		(nparray) vector which is the nth element of the basis
	'''
	return basis[n]

def S_z(basisvec,j):
	'''
	Emulates the action of the S_z operator on a basis 
	vector so that S_z on ket_1 becomes m*ket_1, where
	m is the spin of that vector (1/2 or -1/2). Uses 
	units where hbar is 1.
	Args:
		basisvec: (nparray) vector from the basis 
			that S_z is acting on
		j: (integer) element of the vector that S_z is
			acting on
	Returns:
		(list) same vector (nparray) that was input as the 
		first element of a list and the value of spin as the
		second element (float)
	'''
	if basisvec[j] == 1:
		m = 1/2
	else:
		m = -1/2
	return [basisvec,m]
	
def S_plusminus(basisvec,j,k):
    '''
    Emulates the action of the S_plus*S_minus operator on a 
    basis vector so that S_plus on spin down (0) becomes spin 
    up (1) and S_minus on spin up (1) becomes spin down (0), 
	with units where hbar is 1. Attempting to act S_plus on a 
	spin up (1) state yields numeric 0 and attempting to act 
	S_minus on a spin down (0) state yields numeric 0. 
    Args:
        basisvec: (nparray) vector from basis that S_plus is 
            acting on
        j: (integer) element of the vector that S_plus is 
            acting on
	    k: (integer) element of the vector that S_minus is
			acting on
    Returns:
        (list) same vector (nparray) that was input, except
        with its jth element incremented from 0 to 1 and kth
		element decremented from 1 to 0, or numeric zero if 
		the jth element was already 1 or kth element was already
		0 as the first element of a list, and the index (integer) 
		of the new vector in the basis set as the second element
		in a list
    '''
    if type(basisvec) == int or type(basisvec) == float:
        return [0,0]
    elif basisvec[j] == 1:
        return [0,0]
    elif basisvec[k] == 0:
        return [0,0]
    else:
        vec = copy.deepcopy(basisvec)
        vec[j] = 1
        vec[k] = 0
    return [vec,basis.tolist().index(list(vec))]

def in_prod(vec1,vec2):
    '''
    Takes the inner product of two basis vectors, invoking
    orthonormality. That is, if the two vectors are the same,
    the result will be one; otherwise, the result is 0. This 
    behaves exactly like the Kronecker delta.
    Args:
        vec1: (nparray) vector from basis 
        vec2: (nparray) vector from basis
    Returns:
        (integer) 1 or 0 depending on whether or not the inputs
            are the same or not, respectively
    '''
    if type(vec1) == int or type(vec1) == float:
        return 0
    elif type(vec2) == int or type(vec2) == float:
        return 0
    elif np.array_equal(vec1,vec2) == True:
        return 1
    else:
        return 0
    return

## Generate the Hamiltonian matrix

H = np.zeros((2**n,2**n))

for i in range(2**n):
    if i == 0:
        continue
    for k in range(n-1):
        H[i][S_plusminus(vec(i),k-1,k)[1]] += 0.5
        H[i][S_plusminus(vec(i),k,k+1)[1]] += 0.5
        H[i][S_plusminus(vec(i),k+1,k)[1]] += 0.5
		
for i in range(2**n):              
    for k in range(n):
        H[i][i] += (S_z(vec(i),k)[1]*S_z(vec(i),k-1)[1]*in_prod(vec(i),S_z(vec(i),k)[0])
        *in_prod(vec(i),S_z(vec(i),k-1)[0]))

for i in range(2**n):
    for j in range(2**n):
        if j > i:
            H[j][i] = np.conj(H[i][j])
                
		
print('The Hamiltonian matrix is: ')
print(H)
# end2 = time.clock()
# print('Finding the Hamiltonian took',(end2-end1),'seconds')

## Compute eigenvalues and corresponding eigenvectors

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










	
	
