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


def build_run_path(config, config_path):
    env = config["env"]
    model = config["model"]

    base = Path("runs")
    reward = f"reward_{env['reward_type']}"
    controller = f"controller_{env['controller']}"
    obs = f"observation_{env['observation_type']}"
    
    domain_length = env["L"]
    gamma = model["gamma"]
    buffer = model["buffer_size"]
    lim = env["lim"]

    leaf = f"L_{domain_length}_gamma_{gamma}_buffer_{buffer}_lim_{lim}"

    path = base / reward / controller / obs / leaf
    path.mkdir(parents=True, exist_ok=True)

    # Save config file for reproducibility
    shutil.copy(config_path, path / Path(config_path).name)

    return path


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
            while not terminated[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated = self.eval_env.step(action)
                total_reward += reward[0]
                steps += 1
            wandb.log({
                "eval/mean_reward": total_reward/steps,
                "eval/final_reward": reward[0],
                "global_step": self.num_timesteps
            })
        return True
    

def main(config_path="config_sac.yaml"):
    # Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    save_dir = build_run_path(config, config_path)
    
    env_args = config["env"]
    model_args = config["model"]
    reward = f"reward_{env_args['reward_type']}"
    controller = f"controller_{env_args['controller']}"
    obs = f"observation_{env_args['observation_type']}"
    
    domain_length = env_args["L"]
    gamma = model_args["gamma"]
    buffer = str(int(int(model_args["buffer_size"])/1000))
    lim = env_args["lim"]


    # Init Weights & Biases
    wandb.init(
        project=config["logger"]["project_name"],
        name=f"{reward}_{controller}_{obs}_L_{domain_length}_gamma_{gamma}_buf_{buffer}k_lim_{lim}",
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
        seed=cast_scalar(config["model"].get("seed", None)),
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
        suffix = "interrupted" if interrupted else "final"
        save_path = save_dir / f"sac_ks_{suffix}.zip"
        model.save(str(save_path))
        wandb.save(str(save_path))

    if config["logger"].get("save_replay_buffer", False):
        model.save_replay_buffer(str(save_dir / "sac_ks_replay_buffer.pkl"))
        wandb.save(str(save_dir / "sac_ks_replay_buffer.pkl"))                                      

if __name__ == "__main__":
    config_path = "config_sac.yaml"
    main(config_path)
