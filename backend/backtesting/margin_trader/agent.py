"""DRL Agent wrapper for Stable Baselines 3."""

from __future__ import annotations

import logging

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

MODELS = {"a2c": A2C}


class TensorboardCallback(BaseCallback):
    """Callback for logging rewards to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            reward = self.locals.get("rewards", self.locals.get("reward", [0]))[0]
            self.logger.record(key="train/reward", value=reward)
        except (KeyError, IndexError):
            pass
        return True


class DRLAgent:
    """Wrapper for DRL model training and inference.

    Currently supports A2C algorithm from Stable Baselines 3.
    """

    def __init__(self, env):
        """Initialize agent with environment.

        Args:
            env: Gymnasium-compatible environment
        """
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
        """Create and configure a DRL model.

        Args:
            model_name: Algorithm name (currently only "a2c")
            policy: Policy architecture
            policy_kwargs: Policy configuration
            model_kwargs: Model hyperparameters
            verbose: Logging verbosity
            seed: Random seed
            tensorboard_log: TensorBoard log directory

        Returns:
            Configured SB3 model

        Raises:
            NotImplementedError: If model_name not supported
        """
        if model_name not in MODELS:
            raise NotImplementedError(f"Model '{model_name}' not implemented")

        model_kwargs = model_kwargs or {}

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
        """Train the model with TensorBoard logging.

        Args:
            model: SB3 model to train
            tb_log_name: TensorBoard run name
            total_timesteps: Training steps

        Returns:
            Trained model
        """
        return model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )

    @staticmethod
    def DRL_prediction(model, environment, deterministic: bool = True):
        """Run model inference on environment.

        Args:
            model: Trained SB3 model
            environment: Trading environment
            deterministic: Use deterministic actions

        Returns:
            Tuple of (account_memory, actions_memory, state_memory)
        """
        test_env, test_obs = environment.get_sb_env()
        account_memory = []
        actions_memory = []
        state_memory = []

        test_env.reset()

        for i in range(len(environment.dates)):
            action, _ = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)

            # Collect memories on second-to-last step
            if i == len(environment.dates) - 2:
                account_memory = test_env.env_method("save_asset_memory")
                actions_memory = test_env.env_method("save_action_memory")
                state_memory = test_env.env_method("save_state_memory")

            if dones[0]:
                logger.debug("Episode complete at step %d", i)
                break

        return account_memory[0], actions_memory[0], state_memory[0]
