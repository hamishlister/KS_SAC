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
from policies import LinearSACPolicy, MySACPolicy
from callbacks import WandbEvalCallback

from KSEnv import KS_Env

torch.set_default_dtype(torch.float32)

POLICY_LOOKUP = {
    "MlpPolicy": "MlpPolicy",
    "SACPolicy": MySACPolicy,
    "LinearSACPolicy": LinearSACPolicy
}


def make_env(env_config, seed=None):
    def _init():
        try:
            # Copy and override the seed safely
            local_env_config = dict(env_config)  # shallow copy
            if seed is not None:
                local_env_config["seed"] = seed

            env = KS_Env(**local_env_config)
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

    ens_seed = cast_scalar(config["env"].get("seed", None))
    if ens_seed is not None:
        ens_seed += ens_idx  # Ensure unique seed for each ensemble member

    env_fns = [
        make_env(config["env"], seed=ens_seed * 100 + i)  # 100 spacing to prevent overlap
        for i in range(num_train_envs)
    ]
    env = SubprocVecEnv(env_fns)
    
    eval_env_fns = [
        make_env(config["env"], seed=ens_seed * 1000 + 500 + i)
        for i in range(config["logger"]["num_eval_envs"])
    ]
    eval_env = DummyVecEnv(eval_env_fns)

    # Build model
    policy_kwargs = config["model"].get("policy_kwargs", {})
    print(f"Using policy: {config['model']['policy']} with kwargs: {policy_kwargs}")
    if "activation_fn" in policy_kwargs:
        policy_kwargs["activation_fn"] = resolve_activation(policy_kwargs["activation_fn"])

    # If controller is 'fully_linear', override actor network
    if config["env"].get("controller") == "fully_linear":
        if isinstance(policy_kwargs.get("net_arch"), dict):
            policy_kwargs["net_arch"]["pi"] = []
        else:
            policy_kwargs["net_arch"] = {"pi": [], "qf": [256, 256]}  # default qf fallback if needed

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

    print("Model structure:")
    print(model.policy)

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
    config_path = "configs/config_sac.yaml"
    main(config_path)

  
#     # Early stopping variables
#     low_eval_counter = 0
#     reward_threshold = -10  # Set your threshold
#     max_low_evals = 2

#     # Custom training loop for early stopping
#     total_timesteps = config["train"]["total_timesteps"]
#     callback = WandbEvalCallback(eval_env, eval_freq=eval_step_freq)
#     timesteps = 0
#     interrupted = False
#     while timesteps < total_timesteps:
#         try:
#             # Determine how many steps to run until next eval or end
#             steps_to_run = min(eval_step_freq, total_timesteps - timesteps)
#             model.learn(
#                 total_timesteps=timesteps + steps_to_run,
#                 reset_num_timesteps=False,
#                 callback=callback
#             )
#             timesteps += steps_to_run
#         except KeyboardInterrupt:
#             print("\nTraining interrupted by user. Saving model...")
#             interrupted = True
#             break

#         # Get the latest Wandb logs
#         logs = wandb.run.history._data[-1] if hasattr(wandb.run, "history") and wandb.run.history._data else {}
#         # Early stopping logic
#         if "mean_eval/mean_reward" in logs:
#             if logs["mean_eval/mean_reward"] > reward_threshold:
#                 low_eval_counter += 1
#                 if low_eval_counter >= max_low_evals:
#                     print(f"Early stopping: {max_low_evals} evaluations below threshold")
#                     break  # Exit the training loop
#             else:
#                 low_eval_counter = 0
    
#     # Save model whether completed or interrupted
#     if config["train"].get("save_model", True):
#         save_path = Path("runs") / f"{exp_name}.zip"
#         model.save(str(save_path))
#         wandb.save(str(save_path))

#     if config["logger"].get("save_replay_buffer", False):
#         replay_path = Path("runs") / f"{exp_name}_replay_buffer.pkl"
#         model.save_replay_buffer(str(replay_path))
#         wandb.save(str(replay_path))

# if __name__ == "__main__":
#     config_path = "configs/config_nonlin_time.yaml"
#     main(config_path)
