# wedlm_train - WeDLM PostTraining 核心训练库
# 提供 SFT / DPO / GSPO 训练所需的全部模块

from .config import (
    BaseTrainingConfig,
    SFTConfig,
    DPOConfig,
    GSPOConfig,
    from_yaml,
)
from .data import (
    WeDLMPackedDataset,
    WeDLMShuffledPackedDataset,
    WeDLMPairwiseDataset,
    WeDLMPromptDataset,
    packed_collate_fn,
    dpo_collate_fn,
    gspo_prompt_collate_fn,
    get_im_end_token_id,
)
from .batch import (
    WeDLMBatch,
    build_wedlm_batch,
)
from .model import (
    wedlm_forward,
    wedlm_attention_forward,
    apply_rotary_pos_emb,
)
from .attention import (
    get_attention_wrapper,
    check_backend_available,
    get_available_backend,
)
from .loss import (
    compute_mlm_loss,
    compute_ar_loss,
    compute_block_scores,
    compute_dpo_loss,
    compute_gspo_loss,
    compute_masked_token_logps,
)
from .reward import (
    build_reward_function,
    RewardInputs,
    BaseRewardFunction,
)
from .utils import setup_logger, set_seed, get_device
