from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np



class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            print(f"Evaluating at step {self.num_timesteps}...")
            obs = self.eval_env.reset()
            total_reward = 0.0
            steps = 0
            total_norm = 0.0
            total_action = 0.0
            total_time = 0.0

            terminated = [False] * self.eval_env.num_envs

            u_prev = 0.0
            a_prev = 0.0
            t_prev = 0.0
            r_prev = 0.0

            while not any(terminated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, terminated, truncated = self.eval_env.step(action)

                r_mean = rewards.mean()
                a_norms = self.eval_env.get_attr("forcing_norm")
                u_norms = self.eval_env.get_attr("u_current_norm")
                t_norms = self.eval_env.get_attr("u_t_norm")
                u_mean = np.mean(u_norms)
                a_mean = np.mean(a_norms)
                t_mean = np.mean(t_norms)

                total_norm += u_mean
                total_reward += r_mean
                total_action += a_mean
                total_time += t_mean
                steps += 1

                u_prev = u_mean
                a_prev = a_mean
                t_prev = t_mean
                final_true_reward = - t_prev/u_prev - a_prev

            if hasattr(self.model, "log_ent_coef"):
                alpha = self.model.log_ent_coef.exp().item()
            else:
                alpha = None  # can't find it
                

            log_data = {
                "mean_eval/mean_reward": (total_reward / steps),
                "final_eval/final_reward": r_mean,
                "mean_eval/mean_u_norm": (total_norm / steps),
                "final_eval/final_u_norm": u_mean,
                "mean_eval/mean_action_norm": (total_action / steps),
                "final_eval/final_action_norm": a_mean,
                "mean_eval/mean_time_derivative": (total_time / steps),
                "final_eval/final_time_derivative": t_mean,
                "final_eval/final_true_reward": final_true_reward,
                "global_step": self.num_timesteps,
                "mean_eval/steps": steps
            }
            if alpha is not None:
                log_data["mean_eval/alpha"] = alpha
            wandb.log(log_data)
        return True