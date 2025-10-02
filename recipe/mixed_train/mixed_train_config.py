from dataclasses import dataclass, field

from verl.workers.config import FSDPActorConfig, RolloutConfig


@dataclass
class MixedTrainActorConfig(FSDPActorConfig):
    use_off_policy_loss: bool = False
    off_policy_normalize: bool = False
    off_policy_cliprange: float = 0.2
    use_off_policy_probs: bool = False
    use_off_policy_clip: bool = False
    off_policy_max_clip: float = -1.0
    off_policy_min_clip: float = -1.0
    off_policy_reshape: str = "no_reshape"
    off_policy_reshape_weight: float = 0.1
    off_policy_reshape_pow_exp: float = 0.5
    on_policy_reshape: str = "no_reshape"
    on_policy_reshape_weight: float = 0.1
    on_policy_reshape_pow_exp: float = 0.5
    calculate_sft_loss: bool = True
    all_max_clip: float = -1
    loss_remove_token_mean: bool = False
    loss_remove_clip: bool = False
    sft_loss_coef: float = 0.1 # 仅在calculate_sft_loss为True时生效
    calculate_rl_loss: bool = True

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        super().validate(n_gpus, train_batch_size, model_config)

@dataclass
class MixedTrainRolloutConfig(RolloutConfig):
    n_off_policy: int = 0
    self_explain: dict = field(default_factory=lambda: {"max_tokens": 100, "max_blocks_num": 5})
    chunk_size: int = 8