import yaml
import wandb
import torch
import numpy as np
import os
import torch.nn as nn
import importlib
from pathlib import Path
import shutil
import multiprocessing
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from policies import LinearSACPolicy

from KSEnv import KS_Env

torch.set_default_dtype(torch.float32)

POLICY_LOOKUP = {
    "MlpPolicy": "MlpPolicy",
    "LinearSACPolicy": LinearSACPolicy
}


def make_env(env_config):
    def _init():
        try:
            env = KS_Env(**env_config)
            return env
        except Exception as e:
            print(f"[Worker crash] Could not create env: {e}")
            import traceback; traceback.print_exc()
            raise
    return _init

def cast_scalar(value):
    if isinstance(value, str):
        return value  # e.g., 'auto_0.1' or 'auto'
    elif isinstance(value, (np.generic, np.float64, np.int64)):
        return value.item()
    return value

def resolve_activation(name):
    # Convert string like 'nn.ReLU' to actual nn.ReLU class
    if isinstance(name, str) and name.startswith("nn."):
        return getattr(nn, name.split(".")[1])
    return name  # If it's already a class or something else valid



class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            print(f"Evaluating at step {self.num_timesteps}...")
            obs = self.eval_env.reset()
            terminated = [False]
            total_reward = 0
            steps = 0
            total_norm = 0
            while not terminated[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated = self.eval_env.step(action)
                u_norm = np.linalg.norm(self.eval_env.get_attr("u_current"))
                total_norm += u_norm
                total_reward += reward[0]
                steps += 1
            wandb.log({
                "eval/mean_reward": total_reward/steps,
                "eval/final_reward": reward[0],
                "eval/mean_u_norm": total_norm/steps,
                "eval/final_u_norm": u_norm,
                "global_step": self.num_timesteps
            })
        return True
    

def main(config_path="config_sac.yaml", ens_idx=0):
    # Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_name = f"{str(config_path).split('/')[-1].replace('config_', '').replace('.yaml', '')}_ens{ens_idx}"

    # Init Weights & Biases
    wandb.init(
        project=config["logger"]["project_name"],
        name=exp_name,
        config=config,
        mode=config["logger"].get("mode", "online"),
        group=config["logger"].get("team_name", None)
    )

    # Vectorized environments
    if config["env"]["device"] == "cpu":
        num_cpus = multiprocessing.cpu_count()
        print(f"Detected {num_cpus} CPUs on this machine.")
        num_train_envs = min(config["logger"].get("num_train_envs", 1), num_cpus)
    else:
        num_train_envs = config["logger"].get("num_train_envs", 1)

    print(f"Using {num_train_envs} training environments on device {config['env']['device']}.")
    env_fns = [make_env(config["env"]) for _ in range(num_train_envs)]
    env = SubprocVecEnv(env_fns)

    eval_env_fns = [make_env(config["env"]) for _ in range(config["logger"]["num_eval_envs"])]
    eval_env = SubprocVecEnv(eval_env_fns)

    # Build model
    policy_kwargs = config["model"].get("policy_kwargs", {})
    if "activation_fn" in policy_kwargs:
        policy_kwargs["activation_fn"] = resolve_activation(policy_kwargs["activation_fn"])

    # If controller is 'fully_linear', override actor network
    if config["env"].get("controller") == "fully_linear":
        if isinstance(policy_kwargs.get("net_arch"), dict):
            policy_kwargs["net_arch"]["pi"] = []
        else:
            policy_kwargs["net_arch"] = {"pi": [], "qf": [256, 256]}  # default qf fallback if needed

    ens_seed = cast_scalar(config["env"].get("seed", None))
    if ens_seed is not None:
        ens_seed += ens_idx  # Ensure unique seed for each ensemble member

    model = SAC(
        policy=POLICY_LOOKUP[config["model"]["policy"]],
        env=env,
        learning_rate=cast_scalar(config["model"]["learning_rate"]),
        buffer_size=cast_scalar(config["model"]["buffer_size"]),
        learning_starts=cast_scalar(config["model"]["learning_starts"]),
        batch_size=cast_scalar(config["model"]["batch_size"]),
        tau=cast_scalar(config["model"]["tau"]),
        gamma=cast_scalar(config["model"]["gamma"]),
        train_freq=cast_scalar(config["model"]["train_freq"]),
        gradient_steps=cast_scalar(config["model"]["gradient_steps"]),
        ent_coef=cast_scalar(config["model"]["ent_coef"]),
        action_noise=None,
        policy_kwargs=policy_kwargs,
        verbose=config["model"].get("verbose", 1),
        tensorboard_log=config["model"].get("tensorboard_log", None),
        seed=ens_seed,
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    )

    eval_step_freq = config["logger"]["eval_iter"] * config["env"]["max_steps"]
    print(f"Evaluation every {eval_step_freq} steps.")
    
    # Train
    try:
        model.learn(
            total_timesteps=config["train"]["total_timesteps"],
            callback=WandbEvalCallback(eval_env, eval_freq=eval_step_freq)
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        interrupted = True
    else:
        interrupted = False
    
    # Save model whether completed or interrupted
    if config["train"].get("save_model", True):
        save_path = Path("runs") / f"{exp_name}.zip"
        model.save(str(save_path))
        wandb.save(str(save_path))

    if config["logger"].get("save_replay_buffer", False):
        replay_path = Path("runs") / f"{exp_name}_replay_buffer.pkl"
        model.save_replay_buffer(str(replay_path))
        wandb.save(str(replay_path))                                      

if __name__ == "__main__":
    config_path = "config_sac.yaml"
    main(config_path)
