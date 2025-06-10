import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

from gmres import gmres
from KS_stepper_np import forward_backward_euler, cn_ab, cn_ab_solver, BEFE
from scipy.optimize import minimize_scalar, minimize




# Define the function F(u) = phi^T(u) - u
def F1(u, wavenumbers, T):
    u_next = BEFE(u, T, wavenumbers)
    return u_next - u

# Define the function F(u) = phi^T(u) - u
def F2(u, dt, wavenumbers, T):
    u_hat = np.fft.rfft(u)
    return np.fft.irfft(cn_ab_solver(u, dt, wavenumbers, T, fourier=True)[0][-1] - u_hat)


# Approximates F'(u)v
def jacobian_vector_product(u, v, epsilon, wavenumbers, T, func=F1):
    if func == F1:
        return (func(u=u + epsilon*v, wavenumbers=wavenumbers, T=T) - func(u=u, wavenumbers=wavenumbers, T=T)) / epsilon
    elif func == F2:
        return (func(u=u + epsilon*v, dt=1e-3, wavenumbers=wavenumbers, T=T) - func(u=u, dt=1e-3, wavenumbers=wavenumbers, T=T)) / epsilon


def newton_search(u0, wavenumbers, max_iter=1000, tol=1e-5, T=5, dt=0.01, func=F1, verbose=False):
    # Set up parameters
    u = u0
    k = 0

    while k < max_iter:
        # Compute F(u)
        if func == F1:
            F_u = func(u, wavenumbers=wavenumbers, T=T)
        elif func == F2:
            F_u = func(u, dt=dt, wavenumbers=wavenumbers, T=T)
        # Compute residual
        beta = np.linalg.norm(F_u)

        if verbose:
            print("k = ", k, "beta = ", beta)

        if beta < tol:
            return u
        
        def my_matvec(x):
            return jacobian_vector_product(u, x, epsilon=1e-10, wavenumbers=wavenumbers, T=T, func=func)
        
        n = u0.shape[0]
        A = sp.sparse.linalg.LinearOperator((n, n), matvec=my_matvec)

        
        x0 = np.zeros(u0.shape[0])
        
        y, _ = gmres(A, F_u, x0=x0, verbose=verbose)
        u = u - y
        k += 1
    
    print("Newton search did not converge")
    return u



# Define the function F(u) = T^{α} (phi^T (u)) - u, where T^α is the shift operator
# First version using Tuckerman's method
def F_shift(u, α, dt, wavenumbers, T):
    u_next = BEFE(u, T, wavenumbers)
    u_next_hat = np.fft.rfft(u_next)
    shift = np.exp(1j * α * wavenumbers)
    u_next_shifted = np.fft.irfft(u_next_hat * shift)
    return u_next_shifted - u

# Second version using time stepping
def F_shift2(u, α, dt, wavenumbers, T):
    u_next_hat = cn_ab_solver(u, dt, wavenumbers, T, fourier=True)[0][-1]
    shift = np.exp(1j * α * wavenumbers)
    u_next_shifted = np.fft.irfft(u_next_hat * shift)
    return u_next_shifted - u

# Approximates F'(u)v
def jacobian_vector_product_shift(u, v, α, epsilon, wavenumbers, T, func=F_shift):
    return (func(u=u + epsilon*v, α=α, dt=0.01, wavenumbers=wavenumbers, T=T) - func(u=u, α=α, dt=0.01, wavenumbers=wavenumbers, T=T)) / epsilon

# Approximates dF/dα
def dF_dα(u, α, dt, wavenumbers, T):
    u_hat_next = cn_ab_solver(u, dt, wavenumbers, T, fourier=True)[0][-1]
    shift = np.exp(1j * α * wavenumbers)
    u_hat_shift = u_hat_next * shift
    u_hat_α = (1j * α * wavenumbers) * u_hat_shift
    u_α = np.fft.irfft(u_hat_α)
    return u_α



def linesearch(u, du, α, dα, func=F_shift2, wavenumbers=None, T=None, dt=None):
    def residual_norm(λ):
        λ1, λ2 = λ
        u_trial = u + λ1 * du
        α_trial = α + λ2 * dα
        return np.linalg.norm(func(u_trial, α_trial, dt=dt, wavenumbers=wavenumbers, T=T))
    result = minimize(residual_norm, x0=(1,1))
    λ_opt1, λ_opt2 = result.x
    u_opt = u + λ_opt1 * du
    α_opt = α + λ_opt2 * dα
    return u_opt, α_opt
 



def tw_search(u0, α0, wavenumbers, max_iter=1000, tol=1e-5, T=5, dt=0.01, func=F_shift2, verbose=False):
    # Set up parameters
    u0 = np.asarray(u0)    
    u = u0
    α = α0
    N = u0.shape[0]
    X = np.zeros(N+1)
    X[:N] = u0
    X[N] = α0
    k = 0
    eps = 1e-7

    x0 = np.zeros(u0.shape[0]+1)

    while k < max_iter:
        # Compute F(u)
        F_u = func(u, α, dt, wavenumbers, T)
        # Compute residual
        beta = np.linalg.norm(F_u)
        print("k = ", k, "beta = ", beta, "a = ", α)

        if beta < tol:
            return u, α
        
        def my_matvec(z):
            du, dα = z[:N], z[N]
            jvp = jacobian_vector_product_shift(u, du, α, epsilon=eps, wavenumbers=wavenumbers, T=T, func=func)
            u_α = dF_dα(u, α, dt, wavenumbers, T)
            top = jvp + u_α * dα
            u_x = np.fft.irfft((1j * wavenumbers) * np.fft.rfft(u))
            # bottom is dot product of u_x and du
            bottom = np.dot(u_x, du)
            full = np.concatenate((top, [bottom]))
            return full
            
        
        n = u0.shape[0]
        A = sp.sparse.linalg.LinearOperator((n+1, n+1), matvec=my_matvec)

        
        
        
        print("Starting gmres")
        y, _ = gmres(A, np.concatenate((F_u, [0])), x0=x0, verbose=verbose)
        print("gmres done, starting linesearch")

        du, dα = -y[:N], y[N]
        u, α = linesearch(u, du, α, dα, func=func, wavenumbers=wavenumbers, T=T, dt=dt)
        X[:N] = u
        X[N] = α
        k += 1
    
    print("Newton search did not converge")
    return u, α





if __name__ == "__main__":
    N = 32
    L = 12
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    wavenumbers = np.fft.rfftfreq(N, d=L/N) * 2 * np.pi
    T = 10
    dt = 0.01
    u_tw_initial = [ 0.22303911,  0.55585868,  0.92906109,  1.32237171,  1.70118006,
            2.01368982,  2.19374829,  2.17319869,  1.90450025,  1.387285  ,
            0.68482272, -0.08410174, -0.7802351 , -1.29000093, -1.55863167,
        -1.5958529 , -1.45783066, -1.22009017, -0.95465054, -0.71683036,
        -0.54070692, -0.43987268, -0.4107334 , -0.43695852, -0.49452396,
        -0.55688492, -0.59956468, -0.6033498 , -0.55560061, -0.4497804 ,
        -0.28387399, -0.05868149]
    
    # u_list = cn_ab_solver(u_tw_initial, dt, wavenumbers, T, fourier=False)[0]
    # plt.imshow(
    #     u_list,
    #     cmap = 'jet',
    #     aspect = 'auto',
    #     origin = 'lower',
    #     extent = [-L/2, L/2, 0, u_list.shape[0]*dt],
    #     vmin = -3,
    #     vmax = 3
    # )
    # plt.show()
    
    print("Starting Newton search")
    start = time.time()
    u, α = tw_search(u_tw_initial, 2, wavenumbers, max_iter=1000, tol=1e-5, T=5, dt = 5e-3, func=F_shift2)
    elapsed = time.time() - start
    print("Elapsed time: ", elapsed)
    print(α)
    print(u)

    u_list = cn_ab_solver(u, dt, wavenumbers, 100, fourier=False)[0]
    plt.imshow(
        u_list,
        cmap = 'jet',
        aspect = 'auto',
        origin = 'lower',
        extent = [-L/2, L/2, 0, u_list.shape[0]*dt],
        vmin = -3,
        vmax = 3
    )
    plt.show()



# if __name__ == "__main__":
#     # Set up parameters
#     N = 32
#     L = 12
#     x = np.linspace(-L/2, L/2, N, endpoint=False)
#     wavenumbers = np.fft.rfftfreq(N, d=L/N) * 2 * np.pi
#     u_start = np.sin(2 * np.pi * x/L)
#     T = 100
#     dt = 0.01

#     print("Solving for initial condition")
#     u_list = cn_ab_solver(u_start, dt, wavenumbers, T, fourier=False)[0]
#     u0 = u_list[-1]
#     print("Initial condition solved")

#     print("Starting Newton search")
#     start = time.time()
#     u = newton_search(u0, wavenumbers, max_iter=1000, tol=1e-5, T=5, dt = 5e-3, func=F1)
#     elapsed = time.time() - start
#     print("Elapsed time: ", elapsed)
#     print("Newton search finished")
#     print("u = ", u)

#     print("Starting Newton search")
#     start = time.time()
#     u = newton_search(u0, wavenumbers, max_iter=1000, tol=1e-5, T=5, dt = 5e-3, func=F2)
#     elapsed = time.time() - start
#     print("Elapsed time: ", elapsed)
#     print("Newton search finished")
#     print("u = ", u)



        # def my_matvec(z):
        #     du, da = z[:N], z[N]
        #     u_next_hat_1 = cn_ab_solver(u, dt, wavenumbers, T, fourier=True)[0][-1]
        #     u_next_hat_2 = cn_ab_solver(u + eps*du, dt, wavenumbers, T, fourier=True)[0][-1]
        #     shift = np.exp(-1j * a * wavenumbers)
        #     u_next_shifted_1_hat = u_next_hat_1 * shift
        #     u_next_shifted_1 = np.fft.irfft(u_next_shifted_1_hat)
        #     u_next_shifted_2 = np.fft.irfft(u_next_hat_2 * shift)
        #     F_shift_1 = u_next_shifted_1 - u
        #     F_shift_2 = u_next_shifted_2 - (u + eps*du)
        #     jvp = (F_shift_2 - F_shift_1) / eps
        #     u_α = np.fft.irfft((-1j*a*wavenumbers) * u_next_shifted_1_hat)
        #     top = jvp + u_α
        #     u_x = np.fft.irfft((-1j*wavenumbers) * np.fft.rfft(u))
        #     bottom = np.dot(u_x, du)
        #     return np.concatenate((top, np.array([bottom])))

