import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import pickle
import math
from math import log10, floor
import numpy as np


def forward_backward_euler(u_hat, dt, wavenumbers):
    '''
    Take first step in KS using the forward-backward Euler method (required for two step method)
    u_hat: Fourier transform of the solution at the previous time step
    dt: time step
    wavenumbers: wavenumbers in Fourier space
    Returns:
    u_next_hat: Fourier transform of the solution at the next time step
    '''

    # Fourier linear operator
    L_hat = wavenumbers**2 - wavenumbers**4

    # Derivative in Fourier space
    u_hat_x = 1j * wavenumbers * u_hat
    
    # Inverse Fourier transform
    u_x = np.fft.irfft(u_hat_x)
    u = np.fft.irfft(u_hat)

    # Nonlinear term in physical space
    u_nonlin = - u * u_x

    # Fourier transform
    u_nonlin_hat = np.fft.rfft(u_nonlin)

    # Solve linear part
    u_next_hat = (u_hat + dt * u_nonlin_hat) / (1 - dt * L_hat)

    return u_next_hat



def cn_ab(u_hat, u_prev_nonlin, dt, wavenumbers):
    '''
    Takes a step in KS using the Crank-Nicolson Adams-Bashforth method
    u_hat: Fourier transform of the solution at the previous time step
    u_prev_nonlin: nonlinear term at the previous time step
    dt: time step
    wavenumbers: wavenumbers in Fourier space
    Returns:
    u_next_hat: Fourier transform of the solution at the next time step
    u_nonlin: nonlinear term at the next time step
    u_t: time derivative of the solution
    '''

    # Fourier derivative
    u_hat_x = 1j * wavenumbers * u_hat
    
    # Inverse Fourier transform
    u = np.fft.irfft(u_hat)

    # Nonlinear term in physical space
    u_x = np.fft.irfft(u_hat_x)
    u_nonlin = u * u_x
    
    # Adams-Bashforth
    u_nonlin_next = -0.5 * (3 * u_nonlin - u_prev_nonlin)

    # Fourier transform
    u_nonlin_hat = np.fft.rfft(u_nonlin_next)

    # Dealiasing
    dealiasing_mask = np.abs(wavenumbers) < 2/3 * np.max(wavenumbers)
    u_nonlin_hat = u_nonlin_hat * dealiasing_mask

    # Solve linear part
    linear_operator = wavenumbers**2 - wavenumbers**4
    # Crank-Nicolson for linear part
    u_next_hat = ((1 + 0.5 * linear_operator * dt) * u_hat + dt * u_nonlin_hat) / (1 - 0.5 * dt * linear_operator)

    # Time derivative u_t = -u_xx - u_xxxx - u u_x
    u_t = np.fft.irfft(linear_operator * u_hat) - u_nonlin
    
    return u_next_hat, u_nonlin, u_t



def forced_step(u_hat, u_prev_nonlin, dt, wavenumbers, K):
    '''
    Takes a step in KS using the Crank-Nicolson Adams-Bashforth method with forcing
    u_hat: Fourier transform of the solution at the previous time step
    u_prev_nonlin: nonlinear term at the previous time step
    dt: time step
    wavenumbers: wavenumbers in Fourier space
    K: forcing matrix
    Returns:
    u_next_hat: Fourier transform of the solution at the next time step
    u_nonlin: nonlinear term at the next time step
    u_t: time derivative of the solution
    '''

    # Fourier derivative
    u_hat_x = 1j * wavenumbers * u_hat
    
    # Inverse Fourier transform
    u = np.fft.irfft(u_hat)

    # Forcing
    forcing = K @ u
    forcing_hat = np.fft.rfft(forcing)

    # Nonlinear term in physical space
    u_x = np.fft.irfft(u_hat_x)
    u_nonlin = u * u_x
    
    # Adams-Bashforth
    u_nonlin_next = -0.5 * (3 * u_nonlin - u_prev_nonlin)

    # Fourier transform
    u_nonlin_hat = np.fft.rfft(u_nonlin_next)

    # Dealiasing
    dealiasing_mask = np.abs(wavenumbers) < 2/3 * np.max(wavenumbers)
    u_nonlin_hat = u_nonlin_hat * dealiasing_mask
    forcing_hat = forcing_hat * dealiasing_mask

    # Combining nonlinear and forcing terms
    nonlin_total = u_nonlin_hat + forcing_hat

    # Solve linear part
    linear_operator = wavenumbers**2 - wavenumbers**4

    # Time derivative u_t = -u_xx - u_xxxx - u u_x
    u_t = np.fft.irfft(linear_operator * u_hat) - u_nonlin

    # Crank-Nicolson for linear part
    u_next_hat = ((1 + 0.5 * linear_operator * dt) * u_hat + dt * nonlin_total) / (1 - 0.5 * dt * linear_operator)
    
    return u_next_hat, u_nonlin, u_t



# Crank-Nicolson Adams-Bashforth solver
def cn_ab_solver(u, dt, wavenumbers, t_target, fourier=False, u_prev = None):
    # Initialise time list
    t_list = [0, dt]

    # Initial condition
    if u_prev is None:
        u_prev_hat = np.fft.rfft(u)
        u_hat = forward_backward_euler(u_prev_hat, dt, wavenumbers)
        u_prev = np.fft.irfft(u_prev_hat)
    else:
        u_hat = np.fft.rfft(u)
        u_prev_hat = np.fft.rfft(u_prev)
    # List of solutions
    u_hat_list = [u_prev_hat, u_hat]

    # Computing first nonlinear term
    u_prev_hat_x = 1j * wavenumbers * u_prev_hat
    u_prev_x = np.fft.irfft(u_prev_hat_x)
    u_prev_nonlin = u_prev * u_prev_x

    # Time-stepping
    for i in range(int(t_target/dt) - 1):
        u_hat, u_prev_nonlin, _ = cn_ab(u_hat, u_prev_nonlin, dt, wavenumbers)
        u_hat_list.append(u_hat)
        t_list.append(t_list[-1] + dt)
    
    # Inverse Fourier transform to get the solution in physical space
    u_list = [np.fft.irfft(u_hat) for u_hat in u_hat_list]
    if fourier:
        return np.array(u_hat_list), np.array(u_list), np.array(t_list)
    else:
        return np.array(u_list), np.array(t_list)
    

# Crank-Nicolson Adams-Bashforth solver
def forced_solver(u, dt, wavenumbers, t_target, K, fourier=False, u_prev = None):
    # Initialise time list
    t_list = [0, dt]

    # Initial condition
    if u_prev is None:
        u_prev_hat = np.fft.rfft(u)
        u_hat = forward_backward_euler(u_prev_hat, dt, wavenumbers)
        u_prev = np.fft.irfft(u_prev_hat)
    else:
        u_hat = np.fft.rfft(u)
        u_prev_hat = np.fft.rfft(u_prev)
    # List of solutions
    u_hat_list = [u_prev_hat, u_hat]

    # Computing first nonlinear term
    u_prev_hat_x = 1j * wavenumbers * u_prev_hat
    u_prev_x = np.fft.irfft(u_prev_hat_x)
    u_prev_nonlin = u_prev * u_prev_x

    # Time-stepping
    for i in range(int(t_target/dt) - 1):
        u_hat, u_prev_nonlin, _ = forced_step(u_hat, u_prev_nonlin, dt, wavenumbers, K)
        u_hat_list.append(u_hat)
        t_list.append(t_list[-1] + dt)
    
    # Inverse Fourier transform to get the solution in physical space
    u_list = [np.fft.irfft(u_hat) for u_hat in u_hat_list]
    if fourier:
        return np.array(u_hat_list), np.array(u_list), np.array(t_list)
    else:
        return np.array(u_list), np.array(t_list)
    

def BEFE(u, T, wavenumbers):
    '''
    Computes u_{n+1} = (I - T L)^{-1} (I + T N) u_n = (u_n + T N u_n) / (I - T L)
    u: initial condition
    T: time step
    wavenumbers: wavenumbers in Fourier space
    Returns:
    u_next: Fourier transform of the solution at the next time step
    '''

    # Fourier transform
    u_hat = np.fft.rfft(u)
    # Squared u
    u_squared = u**2
    # Fourier transform
    u_squared_hat = np.fft.rfft(u_squared)
    k = wavenumbers
    nonlin_hat = -0.5j * k * u_squared_hat

    # Dealiasing
    dealiasing_mask = np.abs(wavenumbers) < 2/3 * np.max(wavenumbers)
    nonlin_hat = nonlin_hat * dealiasing_mask

    # Fourier linear operator
    L_hat = wavenumbers**2 - wavenumbers**4

    # BEFE step
    u_next_hat = (u_hat + T * nonlin_hat) / (1 - T * L_hat)
    # Inverse Fourier transform
    u_next = np.fft.irfft(u_next_hat)
    return u_next

    

# def forced_step(u_hat, u_prev_nonlin, dt, wavenumbers, K):
#     '''
#     Takes a step in KS using the Crank-Nicolson Adams-Bashforth method with forcing
#     u_hat: Fourier transform of the solution at the previous time step
#     u_prev_nonlin: nonlinear term at the previous time step
#     dt: time step
#     wavenumbers: wavenumbers in Fourier space
#     K: forcing matrix
#     Returns:
#     u_next_hat: Fourier transform of the solution at the next time step
#     u_nonlin: nonlinear term at the next time step
#     u_t: time derivative of the solution
#     '''

#     # Fourier derivative
#     u_hat_x = 1j * wavenumbers * u_hat
    
#     # Inverse Fourier transform
#     u = np.fft.irfft(u_hat)

#     # Forcing
#     forcing = K @ u
#     forcing_hat = np.fft.rfft(forcing)

#     # Nonlinear term in physical space
#     u_x = np.fft.irfft(u_hat_x)
#     u_nonlin = u * u_x
    
#     # Adams-Bashforth
#     u_nonlin_next = -0.5 * (3 * u_nonlin - u_prev_nonlin)

#     # Fourier transform
#     u_nonlin_hat = np.fft.rfft(u_nonlin_next)

#     # Dealiasing
#     dealiasing_mask = np.abs(wavenumbers) < 2/3 * np.max(wavenumbers)
#     u_nonlin_hat = u_nonlin_hat * dealiasing_mask
#     forcing_hat = forcing_hat * dealiasing_mask

#     # Combining nonlinear and forcing terms
#     nonlin_total = u_nonlin_hat + forcing_hat

#     # Solve linear part
#     linear_operator = wavenumbers**2 - wavenumbers**4

#     # Time derivative u_t = -u_xx - u_xxxx - u u_x
#     u_t = np.fft.irfft(linear_operator * u_hat) - u_nonlin

#     # Crank-Nicolson for linear part
#     u_next_hat = ((1 + 0.5 * linear_operator * dt) * u_hat + dt * nonlin_total) / (1 - 0.5 * dt * linear_operator)
    
#     return u_next_hat, u_nonlin, u_t