#The goal is to find eigenvalues and corresponding eigenvectors 
#for given Hamiltonians of systems with n spins

import numpy as np
import basis_class
import time

#user inputs number of spins
n = int(input("Enter number of spins: "))

start = time.clock()

#generate the basis
basis = basis_class.Basis(n)
print('The basis for this spin system is: ')
print(basis)
print('\n')
basis2 = basis.convert()

end1 = time.clock()
print('Making the basis took',(end1-start),'seconds')

#generate the Hamiltonian

H = np.zeros(shape=(2**n,2**n))    #Generate the blank matrix

for i in range(2**n):              #Populate the matrix with appropriate elements
    for j in range(2**n):
        for k in range(n):
            if i == j:
                H[i][j] += (basis.S_z(basis2,j,k)[1]*basis.S_z(basis2,j,k-1)[1]*basis.in_prod(basis.vec2(basis2,i),basis.S_z(basis2,j,k)[0])
                *basis.in_prod(basis.vec2(basis2,i),basis.S_z(basis2,j,k-1)[0]))
            elif i > j:
                H[i][j] += (0.5*basis.in_prod(basis.vec(i),basis.S_plus(basis.S_minus(basis.vec(j),k-1),k))        
                + 0.5*basis.in_prod(basis.vec(i),basis.S_plus(basis.S_minus(basis.vec(j),k),k-1)))
                H[j][i] = np.conj(H[i][j])
						

print('The Hamiltonian matrix is: ')
print(H)
end2 = time.clock()
print('Finding the Hamiltonian took',(end2-end1),'seconds')

#Compute eigenvalues and corresponding eigenvectors

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