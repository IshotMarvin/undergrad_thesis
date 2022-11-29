import copy
import numpy as np

class Basis(list):
	'''
	A class that represents the basis vectors of a multi-spin-1/2
	system as a list, each element of which describes a multi-spin 
	state vector as lists. Each vector element represents the 
	spin of each particle. An entry of 1 indicates spin up 
	and an entry of 0 indicates spin down.
	'''
	def __init__(self,n):
		'''
		Creates the Basis object as a list whose elements
		are each a basis vector, also represented as a list. 
		The number of spins present in the system must be given.
		Args:
			n: (integer) number of spins
		E.g. if there are 2 spins, the basis will be returned as
		[[1,1],[1,0],[0,1],[0,0]]
		'''
		self.n = n

		#form the basis set with the given spin number n 
		for i in range(2**n):   
			j = "{0:b}".format(i)         #convert index to binary (string)
			j = list(j)                   #convert string to list
			while len(j) < n:             #ensure list is proper dimension (number of spins)
				j = [0] + j
			for k in range(len(j)):       #invert the binary values (making spin up 1 and spin down 0)
				if j[k] == "1":
					j[k] = 0
				else:	
					j[k] = 1
			self.append(j)
			
	def convert(self):
		'''
		Converts Basis into np.array
		'''
		vec = copy.deepcopy(self)
		vec = list(vec)
		
		return np.asarray(vec)
		
	def vec(self,n):
		'''
		Picks out a vector which is an element of the basis.
		Args:
			n: (integer) which vector to pick
		Returns:
			(list) vector which is the nth element of the basis
		'''
		return self[n]
	
	def vec2(self,basis,n):
		return basis[n]
		
	def S_z(self,basis,n,j):
		'''
		Emulates the action of the S_z operator on a basis 
		vector so that S_z on ket_1 becomes m*ket_1, where
		m is the spin of that vector (1/2 or -1/2). Uses 
		units where hbar is 1. 
		Args:
			n: (integer) vector from basis that S_z is acting on
			j: (integer) element of the vector that S_z is
				acting on
		Returns:
			(list) same vector that was input as the first element 
			of a list and the value of spin as the
			second element
		'''
		vector = copy.deepcopy(self.vec2(basis,n))
		if vector[j] == 1:
			m = 1/2
		else:
			m = -1/2
		return [vector,m]
		
	def S_plus(self,vec,j):
		'''
		Emulates the action of the S_plus operator on a basis
		vector so that S_plus on spin down (0) becomes spin 
		up (1), with units where hbar is 1. Attempting to act 
		S_plus on a spin up (1) state yields numeric 0. 
		Args:
			vec: (list) vector from basis that S_plus is acting on
			j: (integer) element of the vector that S_plus is 
				acting on
		Returns:
			(list) same vector that was input, except
			with its jth element incremented from 0 to 1, or 
			zero if the element was already 1
		'''
		vector = copy.deepcopy(vec)
		if vector == 0:
			return 0
		elif vector[j] == 1:
			return 0
		else:
			vector[j] = 1
		return vector
		
	def S_minus(self,vec,j):
		'''
		Emulates the action of the S_minus operator on a basis
		vector so that S_minus on spin up (1) becomes spin 
		down (0), with units where hbar is 1. Attempting to 
		act S_minus on a spin down (0) state yields numeric 0. 
		Args:
			vec: (list) vector from basis that S_minus is acting on
			j: (integer) element of the vector that S_minus is 
				acting on
		Returns:
			(list) same vector that was input, except
			with its jth element incremented from 1 to 0, or 
			numeric zero if the element was already 0.
		'''
		vector = copy.deepcopy(vec)
		if vector == 0:
			return 0
		elif vector[j] == 0:
			return 0
		else:
			vector[j] = 0
		return vector
		
	def in_prod(self,vec1,vec2):
		'''
		Takes the inner product of two basis vectors, invoking
		orthonormality. That is, if the two vectors are the same,
		the result will be one; otherwise, the result is 0. This 
		behaves exactly like the Kronecker delta.
		Args:
			vec1: (list) vector from basis 
			vec2: (list) vector from basis
		Returns:
			(integer) 1 or 0 depending on whether or not the inputs
				are the same or not, respectively
		'''
		if type(vec1) == 'int':
			return 0
		elif type(vec2) == 'int':
			return 0
		elif np.array_equal(vec1,vec2) == True:
			return 1
		else:
			return 0
			
		
		