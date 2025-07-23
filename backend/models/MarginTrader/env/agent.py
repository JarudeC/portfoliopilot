from __future__ import annotations

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

MODELS = {"a2c": A2C}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging rewards to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except Exception:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True


class DRLAgent:
    """
    Wrapper for training and inference of DRL models (A2C).
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name: str,
        policy: str = "MlpPolicy",
        policy_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        verbose: int = 1,
        seed: int | None = None,
        tensorboard_log: str | None = None,
    ):
        if model_name not in MODELS:
            raise NotImplementedError(f"Model '{model_name}' not implemented")

        if model_kwargs is None:
            model_kwargs = {}

        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    def train_model(
        self,
        model,
        tb_log_name: str,
        total_timesteps: int = 5_000,
    ):
        """
        Train the model and log to TensorBoard.
        """
        return model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )

    @staticmethod
    def DRL_prediction(model, environment, deterministic: bool = True):
        """
        Backtest the trained model in `environment`, returning:
          - account_memory: list of portfolio values over time
          - actions_memory: list of actions taken
          - state_memory: list of states recorded
        """
        test_env, test_obs = environment.get_sb_env()
        account_memory = []
        actions_memory = []
        state_memory = []

        test_env.reset()
        for i in range(len(environment.dates)):
            action, _ = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)
            # on the second-to-last step, grab all memories
            if i == len(environment.dates) - 2:
                account_memory = test_env.env_method("save_asset_memory")
                actions_memory = test_env.env_method("save_action_memory")
                state_memory = test_env.env_method("save_state_memory")
            if dones[0]:
                print("hit end!")
                break

        return account_memory[0], actions_memory[0], state_memory[0]
