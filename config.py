# config.py
from dataclasses import dataclass, field, asdict
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class EnvConfig:
    """
    用于设置环境变量的配置。
    这些变量会在训练脚本 (train.py) 的早期被设置。
    """
    # Hugging Face 和网络代理配置
    hf_endpoint: Optional[str] = field(default=None, metadata={"help": "Hugging Face端点镜像，例如：'https://hf-mirror.com'。"})
    http_proxy: Optional[str] = field(default=None, metadata={"help": "HTTP代理服务器地址，例如：'http://user:pass@host:port'。"})
    https_proxy: Optional[str] = field(default=None, metadata={"help": "HTTPS代理服务器地址，例如：'http://user:pass@host:port'。"})
    
    # Weights & Biases 相关环境变量
    wandb_project: Optional[str] = field(default="VerilogGRPO-Enhanced", metadata={"help": "Weights & Biases 项目名称。"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Weights & Biases 实体（用户名或团队名）。如果为None，则使用 'wandb login' 时设置的默认实体。"})
    wandb_run_name_prefix: Optional[str] = field(default="enhanced-grpo-run", metadata={"help": "W&B运行名称的前缀。时间戳和关键参数会自动追加。"})
    wandb_disable: bool = field(default=False, metadata={"help": "设置为True以显式禁用W&B日志记录 (会设置 WANDB_DISABLED=true)。"})

    _generated_wandb_run_name: Optional[str] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        print("INFO: Initializing EnvConfig and setting environment variables...")
        if self.hf_endpoint:
            os.environ['HF_ENDPOINT'] = self.hf_endpoint
            print(f"INFO: Set HF_ENDPOINT to {self.hf_endpoint}")
        if self.http_proxy:
            os.environ['http_proxy'] = self.http_proxy
            print(f"INFO: Set http_proxy to {self.http_proxy}")
        if self.https_proxy:
            os.environ['https_proxy'] = self.https_proxy
            print(f"INFO: Set https_proxy to {self.https_proxy}")
        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        if self.wandb_disable:
            os.environ['WANDB_DISABLED'] = "true"
            print("INFO: W&B logging explicitly disabled via EnvConfig (WANDB_DISABLED=true).")
        else:
            if self.wandb_project:
                os.environ['WANDB_PROJECT'] = self.wandb_project
            
            if self.wandb_entity:
                os.environ['WANDB_ENTITY'] = self.wandb_entity
            
            if self.wandb_run_name_prefix:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                self._generated_wandb_run_name = f"{self.wandb_run_name_prefix}-{timestamp}"
                if not os.getenv('WANDB_RUN_NAME'):
                    os.environ['WANDB_RUN_NAME'] = self._generated_wandb_run_name
        
        print(f"INFO: Effective W&B Environment:")
        print(f"  WANDB_PROJECT: {os.getenv('WANDB_PROJECT')}")
        print(f"  WANDB_ENTITY: {os.getenv('WANDB_ENTITY')}")
        print(f"  WANDB_RUN_NAME: {os.getenv('WANDB_RUN_NAME')}")
        print(f"  WANDB_DISABLED: {os.getenv('WANDB_DISABLED')}")
        print(f"  WANDB_API_KEY is SET: {'YES' if os.getenv('WANDB_API_KEY') else 'NO (relies on login/config)'}")

@dataclass
class ScriptConfig:
    """
    Configuration for the training script execution, paths, and non-GRPO model/data parameters.
    """
    # --- Non-default arguments first ---
    model_name_or_path: str = field(metadata={"help": "Path to pretrained BASE model or model identifier from huggingface.co/models for GRPO training."})
    dataset_path: str = field(metadata={"help": "Path to the dataset manifest JSONL file."})

    # --- Arguments with default values ---
    stage1_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapters from the first stage of training. If provided, these will be loaded and training will continue on them."}
    )
    output_dir_base: str = field(default="grpo_verilog_runs_enhanced", metadata={"help": "Base directory for all outputs. A timestamped subdirectory will be created here."})
    
    cache_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "Directory to cache downloaded models and datasets. If None, Hugging Face defaults will be used."}
    )

    # Enhanced LoRA config
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank for GRPO stage. Increased from 8 for better capacity."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha for GRPO stage. Scaled with rank."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout for GRPO stage."})
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of modules to target for LoRA in GRPO stage."}
    )
    
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum sequence length for tokenizer and model."})
    
    # Enhanced callback config
    callback_num_samples: int = field(default=3, metadata={"help": "Number of samples to generate in InferenceCallback."})
    callback_eval_every_n_steps: int = field(default=25, metadata={"help": "Frequency of running InferenceCallback. Reduced for more monitoring."})

    # Enhanced generation parameters
    gen_temperature: float = field(default=0.8, metadata={"help": "Temperature for generation during GRL. Increased for diversity."})
    gen_top_k: int = field(default=50, metadata={"help": "Top-k for generation during GRL."})
    gen_top_p: float = field(default=0.95, metadata={"help": "Top-p (nucleus) for generation during GRL."})
    gen_repetition_penalty: float = field(default=1.1, metadata={"help": "Repetition penalty to avoid repetitive code."})
    gen_length_penalty: float = field(default=1.0, metadata={"help": "Length penalty for generation."})
    
    # Curriculum learning config (Enhanced with dual-layer support)
    enable_curriculum: bool = field(default=True, metadata={"help": "Enable dual-layer curriculum learning strategy."})
    curriculum_type: str = field(default="dual_layer", metadata={"help": "Type of curriculum: 'dual_layer', 'complexity_only', 'level_only'"})
    curriculum_focus_levels: List[str] = field(
        default_factory=lambda: ["basic", "intermediate", "advanced", "expert"],
        metadata={"help": "Dataset levels to include in curriculum learning."}
    )
    curriculum_complexity_emphasis: str = field(
        default="balanced", 
        metadata={"help": "Complexity emphasis: 'simple', 'balanced', 'complex'"}
    )
    
    # Dual-layer curriculum stages (more fine-grained control)
    curriculum_stages: List[Dict[str, Any]] = field(metadata={"help": "Detailed dual-layer curriculum stages configuration."})
    
    # Experience replay config
    enable_experience_replay: bool = field(default=True, metadata={"help": "Enable experience replay buffer."})
    experience_buffer_size: int = field(default=1000, metadata={"help": "Size of experience replay buffer."})
    replay_sample_ratio: float = field(default=0.2, metadata={"help": "Ratio of replay samples in each batch."})
    
    # --- Field initialized in __post_init__ ---
    output_dir: str = field(init=False, metadata={"help": "Full path to the output directory for this run (dynamically created)."})

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name_parts = []
        if self.model_name_or_path:
            safe_model_name_parts.append(self.model_name_or_path.split('/')[-1])
        if self.stage1_adapter_path:
             safe_model_name_parts.append("s1adapted")

        safe_model_name = "_".join(safe_model_name_parts) if safe_model_name_parts else "unknown_model"
        self.output_dir = os.path.join(self.output_dir_base, f"enhanced_grpo_{safe_model_name}_{timestamp}")
        
        if self.cache_dir:
            os.environ['HF_HOME'] = self.cache_dir
            os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
            os.environ['HF_DATASETS_CACHE'] = os.path.join(self.cache_dir, 'datasets')
            os.makedirs(os.environ['HF_DATASETS_CACHE'], exist_ok=True)
            print(f"INFO: HF_HOME and TRANSFORMERS_CACHE set to: {self.cache_dir}")
            print(f"INFO: HF_DATASETS_CACHE set to: {os.environ['HF_DATASETS_CACHE']}")


@dataclass
class EnhancedRewardConfig:
    """
    Enhanced configuration for reward values used in GRPO with multi-objective optimization.
    """
    # Basic compilation rewards
    compilation_success: float = field(default=2.0, metadata={"help": "Base reward for successful compilation. Increased from 1.0."})
    compilation_failure: float = field(default=-8.0, metadata={"help": "Penalty for compilation failure. Increased magnitude."})
    simulation_crash: float = field(default=-4.0, metadata={"help": "Penalty for simulation crash. Increased magnitude."})
    output_parse_error: float = field(default=-2.0, metadata={"help": "Penalty for output parsing error."})
    missing_code_block_penalty: float = field(default=-6.0, metadata={"help": "Penalty if the generated code block is missing."})
    
    # Enhanced functional correctness rewards (non-linear)
    test_pass_base_reward: float = field(default=1.5, metadata={"help": "Base reward per passed test case."})
    test_pass_bonus_multiplier: float = field(default=1.3, metadata={"help": "Multiplier for consecutive test passes (exponential bonus)."})
    max_functional_reward: float = field(default=15.0, metadata={"help": "Max reward for functional correctness. Increased from 10.0."})
    all_tests_passed_bonus: float = field(default=5.0, metadata={"help": "Bonus if all test cases pass. Increased from 2.0."})
    
    # Code quality rewards
    code_efficiency_bonus: float = field(default=2.0, metadata={"help": "Bonus for efficient code (low complexity, good structure)."})
    code_readability_bonus: float = field(default=1.0, metadata={"help": "Bonus for readable, well-structured code."})
    code_complexity_penalty: float = field(default=-1.0, metadata={"help": "Penalty for overly complex code."})
    
    # Robustness rewards
    edge_case_handling_bonus: float = field(default=1.5, metadata={"help": "Bonus for handling edge cases correctly."})
    synthesis_friendly_bonus: float = field(default=1.0, metadata={"help": "Bonus for synthesis-friendly code."})
    
    # Penalty configurations
    timeout_penalty: float = field(default=-3.0, metadata={"help": "Penalty for timeout during simulation."})
    resource_usage_penalty: float = field(default=-0.5, metadata={"help": "Penalty for excessive resource usage."})
    
    # Multi-objective weights
    functional_weight: float = field(default=0.6, metadata={"help": "Weight for functional correctness in total reward."})
    efficiency_weight: float = field(default=0.2, metadata={"help": "Weight for code efficiency in total reward."})
    readability_weight: float = field(default=0.1, metadata={"help": "Weight for code readability in total reward."})
    robustness_weight: float = field(default=0.1, metadata={"help": "Weight for code robustness in total reward."})
    
    # Dynamic reward scaling
    enable_adaptive_scaling: bool = field(default=True, metadata={"help": "Enable adaptive reward scaling based on training progress."})
    reward_scale_factor: float = field(default=1.0, metadata={"help": "Global reward scale factor for fine-tuning."})
    reward_clipping_range: float = field(default=20.0, metadata={"help": "Clip rewards to [-range, +range] to prevent instability."})
    
    def get_scaled_reward(self, base_reward: float, training_step: int = 0) -> float:
        """Apply adaptive scaling and clipping to rewards."""
        scaled_reward = base_reward * self.reward_scale_factor
        
        if self.enable_adaptive_scaling and training_step > 0:
            # Gradually increase reward sensitivity as training progresses
            adaptive_factor = min(1.0 + (training_step / 5000) * 0.1, 1.5)
            scaled_reward *= adaptive_factor
        
        # Clip to prevent instability
        return max(-self.reward_clipping_range, min(self.reward_clipping_range, scaled_reward))


# Backward compatibility alias
RewardConfig = EnhancedRewardConfig