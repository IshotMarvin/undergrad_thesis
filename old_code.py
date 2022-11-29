
# def EntEnt(vec,n_A=1):
#     '''
#     Calculates the Von Neumann entropy
#     of entanglement for a system
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
#     # print(red_rho)
#     # eigVals = nplin.eigh(red_rho)[0]
#     # entent = 0
#     # for i in range(len(eigVals)):
#     #     entent += -eigVals[i]*np.log(eigVals[i])
#     entent = np.trace(red_rho)
#     return entent

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
#             Options are 'A' and 'B'; default is 'A'
#         n_A (int): number of particles in
#             subsystem A; default is 1
#     Returns:
#         (float) entanglement entropy of
#             subsystem A in vec state
#     '''
#     n_B = n - n_A
#     d_A = 2**n_A
#     d_B = 2**n_B

#     basis = np.identity(d_B)
#     basis_full = []
#     for i in range(d_B):
#         basis_full.append(np.kron(np.identity(d_A),basis[:][i]))
    
#     rho = np.outer(vec,vec)
#     rho_reduced = np.zeros((d_A,d_A),dtype=complex)
#     for i in range(len(basis_full)):
#         rho_reduced += np.conj(basis_full[i]).dot(rho).dot(np.transpose(basis_full[i]))
    
#     eigVals = nplin.eig(rho_reduced)[0]
#     entent = 0
#     for i in range(len(eigVals)):
#         if eigVals[i] == 0:
#             entent += 0
#         else:
#             entent += -eigVals[i]*np.log(eigVals[i])
#     return entent.real

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
