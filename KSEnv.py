import gymnasium as gym
from gymnasium import spaces
import numpy as np
from KS_stepper_np import cn_ab_solver, cn_ab, forward_backward_euler, forced_step
import matplotlib.pyplot as plt
from IPython.display import clear_output
from search import newton_search, F1, F2

np.set_printoptions(precision=6)
np.seterr(all='warn')
np_float = np.float32
np_complex = np.complex64

class KS_Env(gym.Env):
    """
    A Gym environment for the Kuramoto-Sivashinsky equation.
    """
    def __init__(
            self,
            L,
            actuator_loss_weight=1.0,
            seed=None,
            device="cpu",
            N=32,
            dt=0.01,
            max_steps=1000,
            u0=None,
            lim = 1,
            plot=False,
            verbose=False,
            info_freq=1000,
            controller = 'linear',
            sees_state = True,
            observation_type = 'state',
            reward_type = 'time',
            pullback_state = False,
            noise = 1.0,
            initial_amp = 0.01,
            continuous=False,
            terminate_thresh=25.0,
    ):
        self.L = L                                                  # Length of the domain
        self.N = N                                                  # Number of grid points
        self.dt = np.float32(dt)                                                # Time step
        self.x = np.linspace(-L/2, L/2, N, endpoint=False, dtype=np_float)          # Spatial grid
        self.wavenumbers = (np.fft.rfftfreq(N, d=L/N) * 2 * np.pi).astype(np.float32)    # Wavenumbers
        self.num_sensors = int(N)                                   # Number of sensors
        self.seed = seed                                            # Random seed
        self.device = device                                        # Device to use (CPU or GPU)
        self.max_steps = max_steps                                  # Maximum number of steps
        self.actuator_loss_weight = np.float32(actuator_loss_weight)            # Actuator loss weight
        self.trajectory = []                                        # Trajectory of the solution
        self.u_t_list = []                                          # List of time derivatives
        self.lim = lim                                              # Limitation for the action space
        self.plot = plot                                            # Whether to plot the trajectory or not
        self.verbose = verbose                                      # Whether to print rewards every 100 steps or not
        self.info_freq = info_freq                                  # Frequency of printing information
        self.controller = controller                                # Controller type (linear or nonlinear)
        self.sees_state = sees_state                                # Whether the agent sees the state or not
        self.observation_type = observation_type                    # Observation type (state or time derivative)
        self.reward_type = reward_type                              # Reward type (time or trivial)
        self.pullback_state = pullback_state                        # Whether to use pullback or not
        self.u0 = u0                                                # Initial condition
        self.noise = np.float32(noise)                                          # Amplitude of Gaussian noise on initial condition
        self.initial_amp = np.float32(initial_amp)                              # Initial amplitude for the random initial condition
        self.continuous = continuous                                # Whether the environment is continuous or not
        self.terminate_thresh = np.float32(terminate_thresh)                  # Threshold for termination

        if self.seed is not None:
            self.seed = int(seed)
            
        np.random.seed(self.seed)

        self.dealiasing_mask = np.abs(self.wavenumbers) < 2/3 * np.max(self.wavenumbers)
        self.alpha = np.float32(0.0)
        

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=-self.lim*np.ones(N, dtype=np_float), high=self.lim*np.ones(N, dtype=np_float), shape=(N,), dtype=np_float)
        
        if self.observation_type == 'state_plus_time':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, self.num_sensors), dtype=np_float)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_sensors,), dtype=np_float)

        self.episode_rewards = []
        self.average_rewards = []
        self.final_rewards = []
        self.total_rewards = []

        self.eval_list = []
        self.u_current = None

        
    def pullback(self, u):
        '''
        Pulls back the state to the slice hyperplane
        '''
        u_hat = (np.fft.rfft(u)).astype(np_complex)
        phi = np.angle(u_hat[1])
        self.alpha = -phi/self.wavenumbers[1]
        shift = (np.exp(1j * self.wavenumbers * self.alpha)).astype(np_complex)
        u_shifted_hat = u_hat * shift
        u_shifted = (np.fft.irfft(u_shifted_hat)).astype(np_float)
        return u_shifted
    
    def shift(self, u, alpha):
        '''
        Shifts a vector u by alpha
        '''
        u_hat = (np.fft.rfft(u)).astype(np_complex)
        shift = (np.exp(1j * self.wavenumbers * alpha)).astype(np_float)
        u_shifted = (np.fft.irfft(u_hat * shift)).astype(np_float)
        return u_shifted

    def compute_quantities(self, u):
        """
        Computes the necessary quantities for the Kuramoto-Sivashinsky equation.
        """
        u_hat = (np.fft.rfft(u)).astype(np_complex)
        u_hat_x = 1j * self.wavenumbers * u_hat
        u_x = (np.fft.irfft(u_hat_x)).astype(np_float)
        u_nonlin = u * u_x
        u_t = (np.fft.irfft(self.linear_operator * u_hat) - u_nonlin).astype(np_float)
        
        return u_hat, u_x, u_nonlin, u_t
    
    def return_observation(self):
        if self.observation_type == 'state':
            if self.pullback_state:
                obs = self.pullback(self.u_current)
            else:
                obs = self.u_current

        elif self.observation_type == 'time_derivative':
            if self.pullback_state:
                obs = self.pullback(self.u_t)
            else:
                obs = self.u_t

        elif self.observation_type == 'state_plus_time':
            if self.pullback_state:
                obs = np.stack((self.pullback(self.u_current), self.pullback(self.u_t)))
            else:
                obs = np.stack((self.u_current, self.u_t))
        
        else:
            obs = np.zeros(self.num_sensors, dtype=np_float)

        return obs



    def reset(self, seed=None, options=None):
        """
        Resets the environment to initial condition (specified or random).
        """
        super().reset(seed=seed, options=options)
        self.current_step = 0

        if self.u0 is not None:
            if isinstance(self.u0, str):
                self.u0 = np.load(self.u0).astype(np.float32)
            noise = (np.random.normal(loc=0, scale=self.noise, size=self.N)).astype(np_float)
            self.u_prev = (self.u0 + noise - np.mean(noise)).astype(np_float)
        
        elif self.continuous and self.u_current is not None:
            # Use the last state as the initial condition
            self.u_prev = self.u_current.copy()
        else:
            # Generate a random initial condition in Fourier space
            rand_fourier_coef = (np.random.normal(0, 1, self.N//2 +1)).astype(np_float)
            self.u_prev = (np.fft.irfft(rand_fourier_coef)).astype(np_float)
            self.u_prev = (self.u_prev - np.mean(self.u_prev) * np.ones(self.N, dtype=np_float)).astype(np_float)
            self.u_prev = (self.initial_amp * self.u_prev/np.linalg.norm(self.u_prev)).astype(np_float)
        

        self.linear_operator = self.wavenumbers**2 - self.wavenumbers**4
        self.u_prev_hat = (np.fft.rfft(self.u_prev)).astype(np_complex)
        self.u_current_hat = forward_backward_euler(self.u_prev_hat, self.dt, self.wavenumbers).astype(np_complex)
        self.u_prev = (np.fft.irfft(self.u_prev_hat)).astype(np_float)

        self.u_prev_hat_x = (1j * self.wavenumbers * self.u_prev_hat).astype(np_complex)
        self.u_prev_x = (np.fft.irfft(self.u_prev_hat_x)).astype(np_float)
        self.u_prev_nonlin = self.u_prev * self.u_prev_x

        self.u_current = (np.fft.irfft(self.u_current_hat)).astype(np_float)

        # Quantities needed for the forced step
        self.u_hat, self.u_x, self.u_nonlin, self.u_t = self.compute_quantities(self.u_current)

        # Reset the trajectory
        self.trajectory = [self.u_prev.copy(), self.u_current.copy()]
        self.trajectory = np.array(self.trajectory, dtype=np_float)

        self.u_t_list = [self.u_t.copy()]
        self.action_list = []
        
        self.episode_rewards = []


        obs = self.return_observation()

        return obs, {}
    

    def compute_forcing(self, action):
        if self.controller == 'unforced':
            forcing =  np.zeros(self.N, dtype=np_float)
        
        elif self.controller == 'linear':
            # Compute the linear forcing matrix K
            K = np.zeros((self.N, self.N), dtype=np_float)
            for i in range(self.N):
                row = np.roll(action, i, axis=0)
                K[i, :] = row
            forcing =  (K @ self.u_current).astype(np_float)

            # self.unsquashed = np.arctanh(np.clip(action, -1 + 1e-6, 1 - 1e-6))

            # return self.unsquashed - np.mean(self.unsquashed) * np.ones(self.N)
        
        else:
            # Compute the nonlinear forcing term
            forcing =  action - np.mean(action, dtype=np_float) * np.ones(self.N, dtype=np_float)

        if self.pullback_state:
            forcing = self.shift(forcing, -self.alpha)

        return forcing.astype(np_float)
    
    def compute_reward(self):
        if self.reward_type == 'time':
            divisor = np.linalg.norm(self.u_current) if np.linalg.norm(self.u_current) != 0 else 1e-8
            return (- (np.linalg.norm(self.u_t)/divisor + self.actuator_loss_weight * np.linalg.norm(self.forcing, ord=2))).astype(np_float)
        elif self.reward_type == 'trivial':
            return (- (np.linalg.norm(self.u_current) + self.actuator_loss_weight * np.linalg.norm(self.forcing, ord=2))).astype(np_float)
        else:
            raise ValueError("Invalid reward type. Choose 'time' or 'trivial'.")

    
    def step(self, action):
        """
        Takes a step in the environment, with the forcing term
        """

        self.forcing = self.compute_forcing(action)
        
        self.forcing_hat = (np.fft.rfft(self.forcing)).astype(np_complex)

        self.u_nonlin_next = (-0.5 * (3 * self.u_nonlin - self.u_prev_nonlin)).astype(np_float)
        self.u_nonlin_hat = (np.fft.rfft(self.u_nonlin_next)).astype(np_complex)

        self.u_nonlin_hat = self.u_nonlin_hat * self.dealiasing_mask
        self.forcing_hat = self.forcing_hat * self.dealiasing_mask

        self.nonlin_total = self.u_nonlin_hat + self.forcing_hat
        
        self.u_next_hat = (((1 + 0.5 * self.linear_operator * self.dt) * self.u_current_hat + self.dt * self.nonlin_total) / (1 - 0.5 * self.dt * self.linear_operator)).astype(np_complex)

        self.u_prev_nonlin = self.u_nonlin.copy()
        self.u_current = (np.fft.irfft(self.u_next_hat)).astype(np_float)

        self.u_current_hat, self.u_x, self.u_nonlin, self.u_t = self.compute_quantities(self.u_current)

        if np.linalg.norm(self.u_current) > self.terminate_thresh:
            
            reward = -1e10
            terminated = True
            truncated = False
            done = True

            self.average_rewards.append(np.mean(self.episode_rewards, dtype=np.float32))
            self.final_rewards.append(np.mean(self.episode_rewards[-100:], dtype=np.float32))
            if self.plot:
                clear_output(wait=True)
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(
                        self.trajectory,
                        cmap = 'jet',
                        aspect = 'auto',
                        origin = 'lower',
                        extent = [-self.L/2, self.L/2, 0, self.trajectory.shape[0]*self.dt],
                        vmin = -3,
                        vmax = 3
                    )
                ax[0].set_title('Exploding solution, environment reset')
                ax[1].plot(self.episode_rewards[100:-100])
                ax[1].set_title('Episode rewards')
                ax[1].set_xlabel('Step')
                ax[1].set_ylabel('Reward')
                plt.show()
        
            reward = -1e10

        else:
            terminated = self.current_step >= self.max_steps

        
        if np.linalg.norm(self.u_current) <= 100:
            
            reward = self.compute_reward()

            self.episode_rewards.append(reward)

        self.current_step += 1
        
        done = terminated
        truncated = False

        if self.verbose and self.current_step % self.info_freq == 0:
                print(f'''Step: {self.current_step}
Reward: {reward:.7f}
u_t: {np.linalg.norm(self.u_t):.4f}
u_current: {np.linalg.norm(self.u_current):.4f}
Action: {np.linalg.norm(self.forcing):.4f}


                      ''')

        # Add the current state to the trajectory
        self.trajectory = np.append(self.trajectory, [self.u_current], axis=0)
        self.u_t_list = np.append(self.u_t_list, [self.u_t], axis=0)

        if self.plot:
            if self.current_step == self.max_steps:
                clear_output(wait=True)
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(
                        self.trajectory,
                        cmap = 'jet',
                        aspect = 'auto',
                        origin = 'lower',
                        extent = [-self.L/2, self.L/2, 0, self.trajectory.shape[0]*self.dt],
                        vmin = -3,
                        vmax = 3
                    )
                ax[0].set_title(f'Final reward: {self.episode_rewards[-1]:.2f}')
                ax[1].plot(self.episode_rewards[100:])
                ax[1].set_title('Episode rewards')
                ax[1].set_xlabel('Step')
                ax[1].set_ylabel('Reward')
                ax[1].set_ylim(min(self.episode_rewards), 0)
                plt.show()

        obs = self.return_observation()


        # Evaluation of episode performance
        if self.current_step == self.max_steps:
            self.average_rewards.append(np.mean(self.episode_rewards, dtype=np_float))
            self.final_rewards.append(np.mean(self.episode_rewards[-100:], dtype=np_float))

            if self.reward_type == 'time':
                # self.eval_list.append(np.mean([np.linalg.norm(u_t) for u_t in self.u_t_list[500:]]))
                self.eval_list.append([(np.linalg.norm(u_t)).astype(np_float) for u_t in self.u_t_list])
                
            elif self.reward_type == 'trivial':
                # self.eval_list.append(np.mean([np.linalg.norm(u) for u in self.trajectory[500:]]))
                self.eval_list.append([(np.linalg.norm(u)).astype(np_float) for u in self.trajectory])
            


        return obs, reward, terminated, truncated, {}
    

    def return_performance(self, threshold=0.01):
        
        episode_stats = []

        for norms in self.eval_list:
            episode_length = len(norms)
            below_thresh = np.array(norms) < threshold
            num_below = np.sum(below_thresh)

            longest_streak = 0
            current_streak = 0
            for b in below_thresh:
                if b:
                    current_streak += 1
                    longest_streak = max(longest_streak, current_streak)
                else:
                    current_streak = 0
            
            try:
                first_below = np.where(below_thresh)[0][0]
            except IndexError:
                first_below = None

            if first_below is not None:
                time_to_first_below = first_below * self.dt
                post_threshold_mean_norm = np.mean(norms[first_below:])
            else:
                time_to_first_below = None
                post_threshold_mean_norm = None


            percentage_below = num_below / episode_length if episode_length > 0 else 0

            lowest_norm = np.min(norms)

            episode_stats.append({
                'percentage_below': percentage_below,
                'longest_streak': int(longest_streak),
                'time_to_first_below': time_to_first_below,
                'post_threshold_mean_norm': post_threshold_mean_norm,
                'lowest_norm': lowest_norm,
                'episode_length': episode_length,
            })
        
        return episode_stats

    

    def evaluate_model(self, model, num_episodes=1):
        """
        Evaluates the model on the environment for a given number of episodes.
        """
        self.eval_trajectory = []
        self.eval_u_t_list = []
        self.eval_action_list = []
        self.eval_rewards = []
        self.newton_distance = []

        for episode in range(num_episodes):
            obs, _ = self.reset()
            done = False
            
            episode_reward = []
            episode_trajectory = []
            episode_u_t_list = []
            episode_action_list = []
            
            while not done:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, _ = self.step(action)
                
                episode_trajectory.append(obs[0])
                episode_u_t_list.append(obs[1])
                episode_action_list.append(action)
                episode_reward.append(reward)
                
                done = terminated or truncated

            if self.reward_type == 'time':
                # run newton search to find difference between the final state and the target state
                final_state = episode_trajectory[-1]
                converged_state = newton_search(final_state, self.wavenumbers)
                distance = np.linalg.norm(final_state - converged_state)
                self.newton_distance.append(distance)


            
            self.eval_rewards.append(episode_reward)
            self.eval_trajectory.append(np.array(episode_trajectory))
            self.eval_u_t_list.append(np.array(episode_u_t_list))
            self.eval_action_list.append(np.array(episode_action_list))
        
        return self.eval_rewards, self.trajectory, self.u_t_list, self.action_list





# class KS_Env(gym.Env):
#     """
#     A Gym environment for the Kuramoto-Sivashinsky equation.
#     """
#     def __init__(
#             self,
#             L,
#             actuator_loss_weight=1.0,
#             seed=None,
#             device="cpu",
#             N=32,
#             dt=0.01,
#             max_steps=1000,
#             u0=None,
#             lim = 1,
#             plot=False,
#             verbose=0,
#             controller = 'linear',
#             sees_state = True,
#             observation_type = 'state',
#             reward = 'time',
#             pullback_state = False,
#             noise = 1.0,
#             initial_amp = 0.01,
#     ):
#         self.L = L                                                  # Length of the domain
#         self.N = N                                                  # Number of grid points
#         self.dt = dt                                                # Time step
#         self.x = np.linspace(-L/2, L/2, N, endpoint=False)          # Spatial grid
#         self.wavenumbers = np.fft.rfftfreq(N, d=L/N) * 2 * np.pi    # Wavenumbers
#         self.num_sensors = int(N)                                   # Number of sensors
#         self.seed = seed                                            # Random seed
#         self.device = device                                        # Device to use (CPU or GPU)
#         self.max_steps = max_steps                                  # Maximum number of steps
#         self.actuator_loss_weight = actuator_loss_weight            # Actuator loss weight
#         self.trajectory = []                                        # Trajectory of the solution
#         self.u_t_list = []                                          # List of time derivatives
#         self.lim = lim                                              # Limitation for the action space
#         self.plot = plot                                            # Whether to plot the trajectory or not
#         self.verbose = verbose                                      # Whether to print rewards every 100 steps or not
#         self.controller = controller                                # Controller type (linear or nonlinear)
#         self.sees_state = sees_state                                # Whether the agent sees the state or not
#         self.observation_type = observation_type                    # Observation type (state or time derivative)
#         self.reward = reward                                        # Reward type (time or trivial)
#         self.pullback_state = pullback_state                        # Whether to use pullback or not
#         self.u0 = u0                                                # Initial condition
#         self.noise = noise                                          # Amplitude of Gaussian noise on initial condition
#         self.initial_amp = initial_amp                              # Initial amplitude for the random initial condition

#         np.random.seed(self.seed)

#         # if self.observation_type == 'state_plus_time':
#         #     self.num_sensors = int(2*N)
#         # else:
#         #     self.num_sensors = int(N)

        
#         self.action_list = []
        

#         # Define the action and observation spaces
#         self.action_space = spaces.Box(low=-self.lim*np.ones(N), high=self.lim*np.ones(N), shape=(N,), dtype=np.float64)
        
#         if self.observation_type == 'state_plus_time':
#             self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, self.num_sensors), dtype=np.float64)
#         else:
#             self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_sensors,), dtype=np.float64)
#         self.alpha = 0.0

#         self.episode_rewards = []

#         self.average_rewards = []
#         self.final_rewards = []

#         self.eval_list = []

        
#     def pullback(self, u):
#         '''
#         Pulls back the state to the slice hyperplane
#         '''
#         u_hat = np.fft.rfft(u)
#         phi = np.angle(u_hat[1])
#         self.alpha = -phi/self.wavenumbers[1]
#         shift = np.exp(1j * self.wavenumbers * self.alpha)
#         u_shifted_hat = u_hat * shift
#         u_shifted = np.fft.irfft(u_shifted_hat)
#         return u_shifted
    
#     def shift(self, u, alpha):
#         '''
#         Shifts a vector u by alpha
#         '''
#         u_hat = np.fft.rfft(u)
#         shift = np.exp(1j * self.wavenumbers * alpha)
#         u_shifted = np.fft.irfft(u_hat * shift)
#         return u_shifted



#     def reset(self, seed=None, options=None):
#         """
#         Resets the environment to initial condition (specified or random).
#         """
#         super().reset(seed=seed, options=options)
#         self.current_step = 0

#         if self.u0 is not None:
#             noise = np.random.normal(loc=0, scale=self.noise, size=self.N)
#             self.u_prev = self.u0 + noise - np.mean(noise)
        

#         else:
#             rand_coef = (np.random.rand(16)*2 - 1) * self.initial_amp
#             self.u_prev = sum(rand_coef[i] * np.sin(2 * np.pi * (i+1) * self.x/self.L) for i in range(16))

#             # zrs = np.zeros(self.N)
#             # ons = np.ones(self.N)
#             # u = np.random.normal(loc=zrs, scale=ons)
#             # u = 0.01 * u
#             # self.u_prev = u - np.mean(u)*ons
        
#         self.linear_operator = self.wavenumbers**2 - self.wavenumbers**4
#         self.u_prev_hat = np.fft.rfft(self.u_prev)
#         self.u_current_hat = forward_backward_euler(self.u_prev_hat, self.dt, self.wavenumbers)
#         self.u_prev = np.fft.irfft(self.u_prev_hat)

#         self.u_prev_hat_x = 1j * self.wavenumbers * self.u_prev_hat
#         self.u_prev_x = np.fft.irfft(self.u_prev_hat_x)
#         self.u_prev_nonlin = self.u_prev * self.u_prev_x

#         self.u_current = np.fft.irfft(self.u_current_hat)

#         # Quantities needed for the forced step
#         self.u_hat_x = 1j * self.wavenumbers * self.u_current_hat
#         self.u_x = np.fft.irfft(self.u_hat_x)
#         self.u_nonlin = self.u_current * self.u_x
#         self.u_t = np.fft.irfft(self.linear_operator * self.u_current_hat) - self.u_nonlin

#         # Reset the trajectory
#         self.trajectory = []
#         self.trajectory.append(self.u_prev.copy())
#         self.trajectory.append(self.u_current.copy())
#         self.trajectory = np.array(self.trajectory)
#         self.u_t_list = []
#         self.u_t_list.append(self.u_t.copy())
#         self.action_list = []
        
#         self.episode_rewards = []

#         # # Burn in
#         # for _ in range(10000):
#         #     self.step(action=np.zeros(self.N))


#         if self.observation_type == 'state':
#             if self.pullback_state:
#                 obs = self.pullback(self.u_current)
#             else:
#                 obs = self.u_current

#         elif self.observation_type == 'time_derivative':
#             if self.pullback_state:
#                 obs = self.pullback(self.u_t)
#             else:
#                 obs = self.u_t

#         elif self.observation_type == 'state_plus_time':
#             if self.pullback_state:
#                 obs = np.stack((self.pullback(self.u_current), self.pullback(self.u_t)))
#             else:
#                 obs = np.stack((self.u_current, self.u_t))
        
#         else:
#             obs = np.zeros(self.num_sensors)

        


#         return obs, {}
    

#     def compute_forcing(self, action):
#         K = np.zeros((self.N, self.N))
#         for i in range(self.N):
#             row = np.roll(action, i, axis=0)
#             K[i, :] = row

#         return K
    


    
#     def step(self, action):
#         """
#         Takes a step in the environment, with the forcing term
#         """


#         if self.controller == 'linear':
#             K = self.compute_forcing(action)
#             self.forcing = K@self.u_current

#         elif self.controller == 'unforced':
#             self.forcing = np.zeros(self.N)

#         else:
#             self.forcing = action - np.mean(action) * np.ones(self.N)

#         if self.pullback_state:
#             self.forcing = self.shift(self.forcing, -self.alpha)
        

#         self.forcing_hat = np.fft.rfft(self.forcing)

#         self.u_nonlin_next = -0.5 * (3 * self.u_nonlin - self.u_prev_nonlin)
#         self.u_nonlin_hat = np.fft.rfft(self.u_nonlin_next)

#         self.dealiasing_mask = np.abs(self.wavenumbers) < 2/3 * np.max(self.wavenumbers)
#         self.u_nonlin_hat = self.u_nonlin_hat * self.dealiasing_mask
#         self.forcing_hat = self.forcing_hat * self.dealiasing_mask

#         self.nonlin_total = self.u_nonlin_hat + self.forcing_hat

        
#         self.u_next_hat = ((1 + 0.5 * self.linear_operator * self.dt) * self.u_current_hat + self.dt * self.nonlin_total) / (1 - 0.5 * self.dt * self.linear_operator)

#         self.u_current_hat = self.u_next_hat.copy()
#         self.u_prev_nonlin = self.u_nonlin.copy()
#         self.u_current = np.fft.irfft(self.u_current_hat)

#         self.u_hat_x = 1j * self.wavenumbers * self.u_current_hat
#         self.u_x = np.fft.irfft(self.u_hat_x)
#         self.u_nonlin = self.u_current * self.u_x
#         self.u_t = np.fft.irfft(self.linear_operator * self.u_current_hat) - self.u_nonlin

#         if np.linalg.norm(self.u_current) > 100:
            
#             reward = -1e10
#             # self.episode_rewards.append(reward)
#             terminated = True
#             truncated = False
#             done = True

#             self.average_rewards.append(np.mean(self.episode_rewards))
#             self.final_rewards.append(np.mean(self.episode_rewards[-100:]))
#             clear_output(wait=True)
#             fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#             ax[0].imshow(
#                     self.trajectory,
#                     cmap = 'jet',
#                     aspect = 'auto',
#                     origin = 'lower',
#                     extent = [-self.L/2, self.L/2, 0, self.trajectory.shape[0]*self.dt],
#                     vmin = -3,
#                     vmax = 3
#                 )
#             ax[0].set_title('Exploding solution, environment reset')
#             ax[1].plot(self.episode_rewards[100:-100])
#             ax[1].set_title('Episode rewards')
#             ax[1].set_xlabel('Step')
#             ax[1].set_ylabel('Reward')
#             plt.show()
        
#             reward = -1e10

#         else:
#             terminated = self.current_step >= self.max_steps

        
#         if np.linalg.norm(self.u_current) <= 100:
            
#             if self.reward == 'time':
#                 reward = - (np.linalg.norm(self.u_t)       + self.actuator_loss_weight * np.linalg.norm(self.forcing, ord=2))**2
#             elif self.reward == 'trivial':
#                 reward = - (np.linalg.norm(self.u_current) + self.actuator_loss_weight * np.linalg.norm(self.forcing, ord=2))**2

#             self.episode_rewards.append(reward)

#         self.current_step += 1
        
#         done = terminated
#         truncated = False

#         if self.verbose == 1:
#             if self.current_step % 100 == 0:
#                 # print("Step: ", self.current_step, "Reward: ", reward, "u_t: ", np.linalg.norm(self.u_t), "u_current: ", np.linalg.norm(self.u_current))
#                 print(f'Step: {self.current_step}, Reward: {reward:.7f}, u_t: {np.linalg.norm(self.u_t):.4f}, u_current: {np.linalg.norm(self.u_current):.2f}')
#                 print("Action: ", action)
#         elif self.verbose == 2:
#             if self.current_step % 1000 == 0:
#                 print(f'Step: {self.current_step}, Reward: {reward:.7f}, u_t: {np.linalg.norm(self.u_t):.4f}, u_current: {np.linalg.norm(self.u_current):.2f}')
#                 print("Action: ", action)

#         # Add the current state to the trajectory
#         self.trajectory = np.append(self.trajectory, [self.u_current], axis=0)
#         self.u_t_list = np.append(self.u_t_list, [self.u_t], axis=0)

#         if self.plot:
#             if self.trajectory.shape[0] == self.max_steps+2:
#                 clear_output(wait=True)
#                 fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#                 ax[0].imshow(
#                         self.trajectory,
#                         cmap = 'jet',
#                         aspect = 'auto',
#                         origin = 'lower',
#                         extent = [-self.L/2, self.L/2, 0, self.trajectory.shape[0]*self.dt],
#                         vmin = -3,
#                         vmax = 3
#                     )
#                 ax[0].set_title(f'Final reward: {self.episode_rewards[-1]:.2f}')
#                 # plt.colorbar(label='u')
#                 ax[1].plot(self.episode_rewards[100:])
#                 ax[1].set_title('Episode rewards')
#                 ax[1].set_xlabel('Step')
#                 ax[1].set_ylabel('Reward')
#                 plt.show()

#         if self.observation_type == 'state':
#             if self.pullback_state:
#                 obs = self.pullback(self.u_current)
#             else:
#                 obs = self.u_current

#         elif self.observation_type == 'time_derivative':
#             if self.pullback_state:
#                 obs = self.pullback(self.u_t)
#             else:
#                 obs = self.u_t

#         elif self.observation_type == 'state_plus_time':
#             if self.pullback_state:
#                 obs = np.stack((self.pullback(self.u_current), self.pullback(self.u_t)))
#             else:
#                 obs = np.stack((self.u_current, self.u_t))
                
#         else:
#             obs = np.zeros(self.num_sensors)


#         # Evaluation of episode performance
#         if self.current_step == self.max_steps:
#             self.average_rewards.append(np.mean(self.episode_rewards))
#             self.final_rewards.append(np.mean(self.episode_rewards[-100:]))

#             if self.reward == 'time':
#                 self.eval_list.append(np.mean([np.linalg.norm(u_t) for u_t in self.u_t_list[500:]]))
                
#             elif self.reward == 'trivial':
#                 self.eval_list.append(np.mean([np.linalg.norm(u) for u in self.trajectory[500:]]))

#             print(self.eval_list)
            


#         return obs, reward, terminated, truncated, {}








        # # Set counter to 0
        # self.current_step = 0

        # # Starting condition
        # if self.u0 is not None:
        #     self.u_prev = u0
        # else:
        #     rand_coef = np.random.rand(5)*2 - 1
        #     self.u_prev = sum(rand_coef[i] * np.sin(2 * np.pi * (i+1) * self.x/self.L) for i in range(5))

        # # Computing first step with forward-backward Euler
        # self.u_prev_hat = np.fft.rfft(self.u_prev)
        # self.u_current_hat = forward_backward_euler(self.u_prev_hat, self.dt, self.wavenumbers)
        # self.u_prev = np.fft.irfft(self.u_prev_hat)
        # self.u_current = np.fft.irfft(self.u_current_hat)

        # # Computing first nonlinear term
        # self.u_prev_hat_x = 1j * self.wavenumbers * self.u_prev_hat
        # self.u_prev_x = np.fft.irfft(self.u_prev_hat_x)
        # self.u_prev_nonlin = self.u_prev * self.u_prev_x
        # done = False

        # # Quantities needed for the forced step
        # self.u_hat_x = 1j * self.wavenumbers * self.u_current_hat
        # self.u_x = np.fft.irfft(self.u_hat_x)
        # self.u_nonlin = self.u_current * self.u_x
        

        # Initialising the trajectory
        # self.trajectory.append(self.u_prev)
        # self.trajectory.append(self.u_current)
        # self.trajectory = np.array(self.trajectory)
