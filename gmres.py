import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


def gmres(A, b, tol=1e-10, max_iter=None, x0 = None, verbose=False):
    
    # List to store norm of residual at each iteration
    norm_list = []
    
    # If max_iter is not specified, set it to the length of b
    N = b.size
    if max_iter is None:
        max_iter = N
    
    # Initialize Q and H
    Q = np.zeros((N, max_iter+1))
    H = np.zeros((max_iter+1, max_iter))
    
    # Set first column of Q to b/||b||
    Q[:, 0] = b / np.linalg.norm(b)

    # Initialize x
    if x0 is not None:
        x = x0
    else:
        x = np.zeros(N)

    # Iterate
    for n in range(max_iter):

        if verbose:
            print('Krylov basis size =', n)
        
        # Arnoldi process
        # Compute the next column of Q and the corresponding elements of H
        v = A(Q[:, n])

        for j in range(n+1):    
            H[j, n] = np.dot(Q[:, j], v)
            v = v - H[j, n] * Q[:, j]
        
        # Normalize v
        H[n+1, n] = np.linalg.norm(v)
        Q[:, n+1] = v / H[n + 1, n]
        # Solve least squares problem y = argmin || Hn y - ||b|| e1 ||
        e1 = np.zeros(n+1)
        e1[0] = 1
        y = np.linalg.lstsq(H[:n+1, :n+1], np.linalg.norm(b)*e1, rcond=None)[0]

        # Compute the current approximation x = Qn y
        x = Q[:, :n+1] @ y
        # Compute the residual
        r = b - A(x)
        r_norm = np.linalg.norm(r)

        if verbose:
            print('Residual Norm:', r_norm)

        # Check for convergence
        if r_norm < tol:
            print('Converged after {} iterations'.format(n))
            break
        n += 1
        norm_list.append(r_norm)

    print('Final Residual Norm:', r_norm)
    return x, norm_list



# Example problem from Trefethen and Bau
# n_it = 200
# A = 2*np.eye(n_it, dtype='float') + 0.5*np.random.randn(n_it, n_it)/np.sqrt(n_it)
# A_mul = lambda x: A @ x
# b = np.random.randn(n_it)
# x, norm_list = gmres(A_mul, b, tol = 1e-6, verbose=True)

# # Should converge at 4^-n
# norm_theoretical = []
# for i in range(len(norm_list)):
#     norm_theoretical.append(norm_list[0] * 4**(-i))

# exact = la.inv(A).dot(b)
# error = la.norm(x - exact) / la.norm(exact)
# print('Error:', error)

# fig = plt.figure(figsize=(10, 6))
# plt.semilogy(norm_list, 'o', label='GMRES Residual Norm')
# plt.semilogy(norm_theoretical, 'k', alpha=0.5, label='Theoretical Convergence')
# plt.xlabel('Iteration')
# plt.ylabel('Residual Norm')
# plt.title('GMRES Convergence')
# plt.legend()
# plt.show()