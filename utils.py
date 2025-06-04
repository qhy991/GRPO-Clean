# enhanced_utils_v2.py - 更新版本，兼容新的数据集格式
import re
import os
import subprocess
import tempfile
import logging
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datasets import Dataset
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizerBase
import torch
import random
from collections import deque
import wandb
from dataclasses import asdict
import time
import datetime
from typing import Optional # Added Optional
from enhanced_curriculum import EnhancedCurriculumManager # Added EnhancedCurriculumManager

# --- Constants for Verilog Parsing ---
THINK_START = "<think>"
THINK_END = "</think>"
CODE_BLOCK_START = "```verilog"
CODE_BLOCK_END = "```"


logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """Experience replay buffer for storing and sampling high-quality training examples."""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        
    def get_buffer_state(self) -> Dict[str, Any]:
        """获取经验回放池的当前状态，用于保存。"""
        # 注意：确保 buffer 中的内容是 JSON 可序列化的
        # 如果经验条目包含复杂对象，你可能需要自定义序列化逻辑或改用 pickle (需谨慎)
        return {
            "buffer_content": list(self.buffer),  # 将 deque 转换为列表
            "max_size": self.max_size
        }

    def load_buffer_state(self, state_dict: Dict[str, Any]):
        """从字典加载经验回放池的状态。"""
        self.max_size = state_dict.get("max_size", self.max_size) # 确保 max_size 也恢复
        self.buffer = deque(state_dict.get("buffer_content", []), maxlen=self.max_size)
        logger.info(f"经验回放池状态已加载。恢复了 {len(self.buffer)} 条经验，最大容量 {self.max_size}。")

    def add_experience(self, prompt: str, completion: str, reward: float, metadata: Dict[str, Any]):
        """Add a new experience to the buffer."""
        experience = {
            "prompt": prompt,
            "completion": completion,
            "reward": reward,
            "metadata": metadata,
            "timestamp": len(self.buffer)
        }
        self.buffer.append(experience)
    
    def get_high_reward_examples(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top-k high reward examples for replay."""
        if len(self.buffer) == 0:
            return []
        
        sorted_buffer = sorted(self.buffer, key=lambda x: x["reward"], reverse=True)
        return sorted_buffer[:min(top_k, len(sorted_buffer))]
    
    def sample_experiences(self, num_samples: int) -> List[Dict[str, Any]]:
        """Sample experiences with preference for higher rewards."""
        if len(self.buffer) == 0:
            return []
        
        # Weighted sampling based on reward
        experiences = list(self.buffer)
        weights = np.array([max(0.1, exp["reward"]) for exp in experiences])  # Ensure positive weights
        weights = weights / weights.sum()  # Normalize
        
        try:
            indices = np.random.choice(len(experiences), size=min(num_samples, len(experiences)), p=weights, replace=False)
            return [experiences[i] for i in indices]
        except:
            # Fallback to uniform sampling if weighted sampling fails
            return random.sample(experiences, min(num_samples, len(experiences)))
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {"size": 0, "mean_reward": 0, "max_reward": 0, "min_reward": 0}
        
        rewards = [exp["reward"] for exp in self.buffer]
        return {
            "size": len(self.buffer),
            "mean_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "std_reward": np.std(rewards)
        }
# 新增步骤：将 enhance_prompt_func 的输出包装进 Qwen 对话模板
def wrap_prompt_for_qwen(example: Dict[str, Any]) -> Dict[str, Any]:
    enhanced_content = example.get("prompt")
    # 构建更严格的Qwen格式，明确要求格式
    strict_format_reminder = """
please write in this format：

<think>
your thinking process
</think>

```verilog
your verilog  code
```
Start answering the question:
"""
    if enhanced_content and isinstance(enhanced_content, str):
        # 保留原始的增强后提示，以防调试或其他地方需要
        example["original_enhanced_prompt"] = enhanced_content
        # 构建 Qwen Instruct 对话格式的提示
        # 注意：这里假设 enhance_prompt_func 的输出适合作为 user 的全部内容
        # 如果 enhance_prompt_func 的输出包含了类似 "system" 的指令，也可以考虑拆分
        example["prompt"] = f"<|im_start|>system\nYou are an Verilog expert。{strict_format_reminder}<|im_end|>\n<|im_start|>user\n{enhanced_content.strip()}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # 如果 prompt 不是字符串或为空，可能需要记录警告或错误
        logger.warning(f"Encountered non-string or empty prompt during Qwen wrapping: {enhanced_content}")
        # 可以选择保留原样，或将其设为一个错误标记，或返回 None 以便后续过滤
        # example["prompt"] = "" # 或者保持原样
    return example

def enhance_prompt_func(example: Dict[str, Any]) -> Dict[str, str]:  
    """  
    Improved prompt enhancement function with stricter formatting requirements  
    """  
    original_prompt = str(example.get("prompt", "No original prompt."))  
    ref_path = str(example.get("reference_verilog_path", ""))  

    module_name, ports = "", []  
    if ref_path and os.path.exists(ref_path):  
        module_name, ports = extract_module_info(ref_path)  
    if not module_name:  
        logger.warning(f"UTILS: EnhancePrompt: Could not extract module from {ref_path}. Using fallback 'generated_module'.")  
        module_name = "generated_module"  

    port_desc_list = []  
    if ports and ref_path and os.path.exists(ref_path):  
        with open(ref_path, "r", encoding="utf-8") as f_ref:  
            ref_content = f_ref.read()  
        for p_name in ports:  
            try:  
                regex = r"(\b(?:input|output|inout)\b\s*(?:(?:reg|wire|logic|signed|unsigned)\s*)?(?:\[[^\]]+\]\s*)?\b" + re.escape(p_name) + r"\b)"  
                match = re.search(regex, ref_content, re.IGNORECASE | re.MULTILINE)  
                if match:  
                    full_decl = re.sub(r'\s+', ' ', match.group(1).strip().replace("\n", " "))  
                    port_desc_list.append(full_decl)  
                else:  
                    port_desc_list.append(f"`{p_name}` (details not auto-detected)")  
            except Exception as e_inner:  
                logger.warning(f"UTILS: EnhancePrompt: Error processing port '{p_name}': {e_inner}")  
                port_desc_list.append(f"`{p_name}` (processing error)")  
    elif ports:  
        port_desc_list = [f"`{p}`" for p in ports]  

    port_desc = ("; ".join(port_desc_list) if port_desc_list else "as implied by the design context")  

    # Stricter formatting guidelines  
    system_instruction = f"""**CRITICAL: You must strictly adhere to the following format. Violations will result in a score of 0!**  
1. First, provide your detailed thought process between <think> and </think> tags.  
2. Then, provide the complete Verilog module code between ```verilog and ```.  
3. Do not include any other content or formatting.  

**Required output format example:**  
<think>  
I need to design a module named {module_name}...  
Step 1: Analyze requirements...  
Step 2: Design the interface...  
Step 3: Implement the logic...  
</think>  
```verilog  
module {module_name}(...);  // Your implementation code  
endmodule  
```  

**Strict requirements:**  
- The module name must be: `{module_name}`  
- Must include all required ports: {port_desc}  
- The code must be complete and synthesizable  
- Must pass all test cases  

**Original problem description:**  
{original_prompt}  

Now, please provide your solution in the strict format specified above:  
"""  

    return {"prompt": system_instruction, "original_prompt_for_debug": original_prompt}


class EnhancedInferenceCallback(TrainerCallback):
    """Enhanced inference callback with better monitoring and logging."""
    
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 eval_dataset: Optional[Dataset] = None,
                 fixed_test_prompts: Optional[List[str]] = None,
                 num_samples: int = 1,
                 eval_every_n_steps: int = 100,
                 max_new_tokens: int = 512,
                 max_seq_length: int = 2048,
                 experience_buffer: Optional[ExperienceBuffer] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.fixed_test_prompts = fixed_test_prompts if fixed_test_prompts is not None else []
        self.num_samples = num_samples
        self.eval_every_n_steps = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.experience_buffer = experience_buffer
        self.generation_history = []

        if not self.fixed_test_prompts and self.eval_dataset is None:
            logger.warning("UTILS: EnhancedInferenceCallback: No fixed prompts and no eval_dataset. Callback may not generate samples.")
# 在 EnhancedInferenceCallback 类内部

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_every_n_steps == 0:
            if model is None:
                logger.warning("UTILS: EnhancedInferenceCallback: Model not provided, skipping inference.")
                return

            if args.local_rank <= 0: # 仅在主进程执行
                logger.info(f"\n--- Enhanced Inference Callback at Step {state.global_step} ---")

                prompts_for_model_generation = [] # 存储最终给模型生成用的、已格式化的prompts
                source_info_for_logging = []      # 存储每个prompt的来源信息，用于日志

                # 1. 处理 self.fixed_test_prompts
                if self.fixed_test_prompts:
                    num_to_sample_fixed = min(self.num_samples, len(self.fixed_test_prompts))
                    if num_to_sample_fixed > 0:
                        sampled_indices_fixed = random.sample(range(len(self.fixed_test_prompts)), num_to_sample_fixed)
                        for orig_idx in sampled_indices_fixed:
                            original_fixed_prompt_content = self.fixed_test_prompts[orig_idx]
                            # 为 fixed_prompts 应用 Qwen 对话格式
                            qwen_formatted_prompt = f"<|im_start|>user\n{original_fixed_prompt_content.strip()}<|im_end|>\n<|im_start|>assistant\n"
                            prompts_for_model_generation.append(qwen_formatted_prompt)
                            source_info_for_logging.append(f"Fixed Prompt #{orig_idx + 1} (Qwen Formatted for model)")

                # 2. 处理 self.eval_dataset (假设其 'prompt' 列已由主流程格式化)
                if self.eval_dataset and len(prompts_for_model_generation) < self.num_samples:
                    num_needed_from_eval = self.num_samples - len(prompts_for_model_generation)
                    if len(self.eval_dataset) > 0 and num_needed_from_eval > 0:
                        num_to_sample_eval = min(num_needed_from_eval, len(self.eval_dataset))
                        eval_indices = random.sample(range(len(self.eval_dataset)), num_to_sample_eval)
                        for dataset_idx in eval_indices:
                            try:
                                # 假设: self.eval_dataset[dataset_idx]['prompt'] 已经是 Qwen 格式化的
                                prompt_from_dataset = self.eval_dataset[dataset_idx]['prompt']
                                prompt_as_str = prompt_from_dataset if isinstance(prompt_from_dataset, str) else str(prompt_from_dataset)

                                # 可选的检查：如果看起来不像 Qwen 格式，则警告
                                if not prompt_as_str.strip().endswith("<|im_start|>assistant\n"):
                                    logger.warning(
                                        f"UTILS: EnhancedInferenceCallback: Prompt from eval_dataset (idx {dataset_idx}) "
                                        f"does NOT look Qwen-formatted. Using as-is. Prompt preview: {prompt_as_str[:100]}..."
                                    )
                                
                                prompts_for_model_generation.append(prompt_as_str)
                                
                                # original_prompt_for_debug 应该追溯到最原始的用户输入，或者至少是 enhance_prompt_func 的输出
                                # 这取决于 eval_dataset 中保存了哪些列
                                orig_debug_prompt = self.eval_dataset[dataset_idx].get('original_prompt_for_debug', 'N/A')
                                if orig_debug_prompt == 'N/A': # 尝试获取 enhance_prompt_func 未包装前的输出
                                    orig_debug_prompt = self.eval_dataset[dataset_idx].get('original_enhanced_prompt', 'N/A')
                                
                                source_info_for_logging.append(f"Eval Dataset Idx {dataset_idx} (Orig: {orig_debug_prompt[:50]}...) (Assumed Qwen Formatted)")

                            except Exception as e:
                                logger.error(f"UTILS: EnhancedInferenceCallback: Error processing eval_dataset sample at index {dataset_idx}: {e}", exc_info=True)
                
                if not prompts_for_model_generation:
                    logger.warning("UTILS: EnhancedInferenceCallback: No prompts available to test for this callback step.")
                    return

                original_training_mode = model.training
                model.eval()

                generation_results_for_wandb = [] # 用于W&B日志的简化结果
                full_generation_history_entries = [] # 用于内部历史记录的详细结果

                for i, text_prompt_for_model_input in enumerate(prompts_for_model_generation):
                    prompt_source_log_msg = source_info_for_logging[i] if i < len(source_info_for_logging) else f"Prompt {i+1}"
                    # 打印给模型实际输入的前缀，确认是Qwen格式
                    logger.info(f"Testing {prompt_source_log_msg} (Model Input Starts With):\n{text_prompt_for_model_input[:150]}...")

                    max_prompt_len = self.max_seq_length - self.max_new_tokens
                    if max_prompt_len <= 0:
                        logger.error(f"UTILS: EnhancedInferenceCallback: max_new_tokens ({self.max_new_tokens}) "
                                     f"is too large for max_seq_length ({self.max_seq_length}). Skipping this prompt.")
                        continue

                    inputs = self.tokenizer(text_prompt_for_model_input, return_tensors="pt", padding=False, truncation=True, max_length=max_prompt_len).to(model.device)
                    
                    try:
                        with torch.no_grad():
                            # --- gen_config 处理 ---
                            base_gen_config = model.generation_config if hasattr(model, "generation_config") and model.generation_config is not None else GenerationConfig()
                            
                            # 确保 tokenizer 的 eos_token_id 是单个 ID
                            tokenizer_eos_id = self.tokenizer.eos_token_id
                            if isinstance(tokenizer_eos_id, list):
                                tokenizer_eos_id = tokenizer_eos_id[0]
                            
                            # 确保 tokenizer 的 pad_token_id 是单个 ID (通常等于 eos_id 如果未特别设置)
                            tokenizer_pad_id = self.tokenizer.pad_token_id
                            if tokenizer_pad_id is None:
                                tokenizer_pad_id = tokenizer_eos_id
                            
                            # 更新 GenerationConfig 实例
                            current_gen_config = GenerationConfig(
                                **base_gen_config.to_dict(), # 从模型的基础配置开始
                                pad_token_id = tokenizer_pad_id,
                                eos_token_id = tokenizer_eos_id, # 使用处理后的tokenizer的eos_id
                                max_new_tokens = self.max_new_tokens,
                                do_sample = True,
                                # 可以从 args 或 script_cfg 添加 temperature, top_p, top_k
                                temperature = getattr(args, 'temperature', getattr(model.config, 'temperature', 0.7)),
                                top_p = getattr(args, 'top_p', getattr(model.config, 'top_p', 0.9)),
                                top_k = getattr(args, 'top_k', getattr(model.config, 'top_k', 50)),
                            )
                            # --- end gen_config 处理 ---

                            outputs = model.generate(
                                **inputs, # 包含 input_ids 和 attention_mask
                                generation_config=current_gen_config
                            )
                        
                        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
                        raw_llm_output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        
                        # 修改这里：传递prompt和调试信息
                        reasoning_part, code_part = parse_llm_completion_with_context(
                            raw_llm_output_text,
                            prompt=text_prompt_for_model_input,  # 传递完整的model输入prompt
                            step=state.global_step,
                            sample_idx=i
                        )
                        reasoning_part_final = reasoning_part if reasoning_part is not None else "REASONING NOT FOUND"
                        code_part_final = code_part if code_part is not None else "CODE NOT FOUND"
                        
                        quality_metrics = assess_code_quality(code_part_final) if code_part_final != "CODE NOT FOUND" else {}
                        
                        # W&B 日志用:
                        generation_results_for_wandb.append({
                            "step": state.global_step,
                            # "prompt_source": prompt_source_log_msg, # W&B 中可能不需要这么详细
                            # "reasoning": reasoning_part_final, # W&B 中可能不需要完整文本
                            # "code": code_part_final,         # W&B 中可能不需要完整文本
                            "quality_metrics": quality_metrics
                        })

                        # 内部历史记录用 (更详细):
                        full_generation_history_entries.append({
                            "step": state.global_step,
                            "prompt_source_info": prompt_source_log_msg,
                            "model_input_prompt": text_prompt_for_model_input, # 保存实际输入给模型的完整prompt
                            "raw_llm_output": raw_llm_output_text,
                            "parsed_reasoning": reasoning_part_final,
                            "parsed_code": code_part_final,
                            "quality_metrics": quality_metrics
                        })
                        
                        if self.experience_buffer and code_part_final != "CODE NOT FOUND":
                            quality_score = sum(quality_metrics.values()) / max(1, len(quality_metrics))
                            # 存入经验池的是给模型训练用的提示 (已Qwen格式化) 和原始输出
                            self.experience_buffer.add_experience(
                                text_prompt_for_model_input, 
                                raw_llm_output_text, 
                                quality_score, 
                                {"step": state.global_step, "source_log_msg": prompt_source_log_msg}
                            )
                        
                        log_output_console = (f"\n--- Generated Reasoning ({prompt_source_log_msg}) ---\n{reasoning_part_final}\n"
                                              f"--- Generated Code ({prompt_source_log_msg}) ---\n{code_part_final}\n"
                                              f"--- Quality Metrics ---\n{quality_metrics}\n--------------------------")
                        logger.info(log_output_console)

                    except Exception as e:
                        logger.error(f"UTILS: EnhancedInferenceCallback: Error during generation for {prompt_source_log_msg}: {e}", exc_info=True)

                # Log generation statistics to W&B if available
                if generation_results_for_wandb and hasattr(wandb, 'run') and wandb.run is not None:
                    try:
                        avg_quality_for_wandb = {}
                        # 从 generation_results_for_wandb 中提取 quality_metrics
                        all_metrics_dicts = [res['quality_metrics'] for res in generation_results_for_wandb if res['quality_metrics']]
                        
                        for metric_name in ['complexity', 'readability', 'efficiency', 'structure']:
                            values = [m.get(metric_name, 0) for m in all_metrics_dicts]
                            if values:
                                avg_quality_for_wandb[f'avg_{metric_name}'] = np.mean(values)
                        
                        wandb.log({
                            f"inference_callback/step": state.global_step,
                            f"inference_callback/num_samples_generated": len(generation_results_for_wandb),
                            **{f"inference_callback/{k}": v for k, v in avg_quality_for_wandb.items()}
                        })
                    except Exception as e_wandb:
                        logger.warning(f"Failed to log inference callback results to W&B: {e_wandb}")

                # Store detailed generation history internally
                self.generation_history.extend(full_generation_history_entries)
                if len(self.generation_history) > 100: 
                    self.generation_history = self.generation_history[-100:]

                if original_training_mode:
                    model.train()
                
                logger.info(f"--- End of Enhanced Inference Callback at Step {state.global_step} ---\n")
# 2. 增强的推理回调，输出生成样本
class DetailedInferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=2, eval_every_n_steps=25, 
                 max_new_tokens=512, max_seq_length=2048, experience_buffer=None, output_dir=None,
                 curriculum_manager: Optional[EnhancedCurriculumManager] = None): # Added curriculum_manager
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset # Fallback
        self.num_samples = num_samples
        self.eval_every_n_steps = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.experience_buffer = experience_buffer
        self.generation_history = []
        self.output_dir = output_dir
        self.curriculum_manager = curriculum_manager # Store curriculum_manager
        
        # 创建生成样本保存目录
        if self.output_dir:
            self.samples_dir = os.path.join(output_dir, "generated_samples")
            os.makedirs(self.samples_dir, exist_ok=True)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_every_n_steps == 0:
            if model is None or args.local_rank > 0: # 确保只在主进程执行
                return

            logger.info(f"\n🔍 === 推理回调 (DetailedInferenceCallback) - 步数 {state.global_step} ===")
            
            eval_dataset_to_use = self.eval_dataset # Fallback to the original dataset

            if self.curriculum_manager and hasattr(self.curriculum_manager, 'is_enabled') and self.curriculum_manager.is_enabled():
                current_stage_dataset = self.curriculum_manager.get_current_stage_dataset()
                if current_stage_dataset and len(current_stage_dataset) > 0:
                    eval_dataset_to_use = current_stage_dataset
                    logger.info(f"DetailedInferenceCallback: Using curriculum stage dataset with {len(eval_dataset_to_use)} samples.")
                else:
                    logger.warning("DetailedInferenceCallback: Curriculum manager enabled, but current stage dataset is empty or None. Using fallback eval_dataset.")
            elif self.curriculum_manager: # Curriculum manager exists but might not be 'enabled' via a specific flag
                current_stage_dataset = self.curriculum_manager.get_current_stage_dataset()
                if current_stage_dataset and len(current_stage_dataset) > 0:
                    eval_dataset_to_use = current_stage_dataset
                    logger.info(f"DetailedInferenceCallback: Using curriculum stage dataset with {len(eval_dataset_to_use)} samples (no explicit is_enabled check or passed).")
                else:
                    logger.warning("DetailedInferenceCallback: Curriculum manager present, but current stage dataset is empty or None. Using fallback eval_dataset.")

            if not eval_dataset_to_use or len(eval_dataset_to_use) == 0:
                logger.warning("DetailedInferenceCallback: No valid dataset available for evaluation. Skipping.")
                return

            # Determine the number of samples to select from eval_dataset_to_use
            num_samples_to_take = min(self.num_samples, len(eval_dataset_to_use))

            if num_samples_to_take > 0:
                sample_indices = random.sample(range(len(eval_dataset_to_use)), num_samples_to_take)
            else:
                logger.warning("DetailedInferenceCallback: Not enough samples to take for evaluation. Skipping.")
                return

            # The original code iterated through sample_indices and used self.eval_dataset[idx]
            # Now it should use eval_dataset_to_use[idx]
            # The rest of the loop structure can remain similar

            # if self.eval_dataset and len(self.eval_dataset) > 0: # This check is now handled by eval_dataset_to_use
            #     sample_indices = random.sample(range(len(self.eval_dataset)),
            #                                    min(self.num_samples, len(self.eval_dataset)))
                
            for i, idx in enumerate(sample_indices):
                sample = eval_dataset_to_use[idx] # Use eval_dataset_to_use
                    
                # 假设: sample['prompt'] 已经是 Qwen 格式化的
                # 这是因为 eval_dataset 应该是由主训练脚本的 full_dataset_processing_pipeline 处理过的
                    prompt_from_dataset = sample['prompt'] 
                    
                    logger.info(f"\n📝 样本 {i+1}/{len(sample_indices)} (数据集索引: {idx})")
                    
                    # 调试日志：打印 prompt 的前缀，检查其格式
                    logger.debug(f"DetailedInferenceCallback: Input prompt for model (first 150 chars): {str(prompt_from_dataset)[:150]}...")

                    # 可选的格式检查和警告
                    if isinstance(prompt_from_dataset, str) and not prompt_from_dataset.strip().endswith("<|im_start|>assistant\n"):
                        logger.warning(
                            f"DetailedInferenceCallback: Prompt from eval_dataset (idx {idx}) "
                            f"does NOT look Qwen-formatted. Using as-is. Prompt preview: {prompt_from_dataset[:100]}..."
                        )
                    elif not isinstance(prompt_from_dataset, str):
                         logger.warning(
                            f"DetailedInferenceCallback: Prompt from eval_dataset (idx {idx}) is not a string (type: {type(prompt_from_dataset)}). "
                            f"Will be cast to string."
                        )
                         prompt_from_dataset = str(prompt_from_dataset) # 确保是字符串

                    # 获取原始问题描述，用于日志记录
                    # 优先使用 original_prompt_for_debug，其次是 original_enhanced_prompt
                    orig_prompt_for_log = sample.get('original_prompt_for_debug', 'N/A')
                    if orig_prompt_for_log == 'N/A':
                        orig_prompt_for_log = sample.get('original_enhanced_prompt', 'N/A')

                    logger.info(f"原始问题 (用于日志): {orig_prompt_for_log[:100]}...")
                    logger.info(f"等级: {sample.get('level', 'unknown')}")
                    logger.info(f"复杂度: {sample.get('complexity_score', 'unknown')}")
                    
                    try:
                        # 传递已经是Qwen格式化的 prompt_from_dataset 给 _generate_single_sample
                        generated_result = self._generate_single_sample(model, prompt_from_dataset, state.global_step)
                        
                        if self.output_dir:
                            self._save_generation_sample(state.global_step, i, sample, generated_result)
                        
                        quality_metrics = assess_code_quality(generated_result.get('code', ''))
                        
                        logger.info(f"生成时间: {generated_result.get('generation_time', 0):.2f}秒")
                        logger.info(f"代码质量: {quality_metrics}")
                        
                        if sample.get('testbench_path') and generated_result.get('code'):
                            test_result = self._run_quick_test(generated_result['code'], sample)
                            logger.info(f"功能测试: {test_result}")
                            
                    except Exception as e:
                        logger.error(f"生成或处理样本 {idx} 失败: {e}", exc_info=True) # 添加 exc_info=True
            
            logger.info(f"🔍 === 推理回调 (DetailedInferenceCallback) 结束 ===\n")
            
    def _generate_single_sample(self, model, prompt, step):
        """生成单个样本"""
        model.eval()
        
        # 准备输入
        max_prompt_len = self.max_seq_length - self.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               truncation=True, max_length=max_prompt_len).to(model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generation_time = time.time() - start_time
        
        # 解码生成文本
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 解析推理和代码
        reasoning, code = parse_llm_completion_with_context(
            generated_text,
            prompt=prompt,  # 传递输入prompt
            step=step,
            sample_idx=0  # 单个样本
        )
        
        model.train()  # 恢复训练模式
        
        return {
            'reasoning': reasoning,
            'code': code,
            'raw_output': generated_text,
            'generation_time': generation_time,
            'step': step
        }
    
    def _save_generation_sample(self, step, sample_idx, original_sample, generated_result):
        """保存生成样本到文件，包含模型实际输入"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 从 original_sample 中提取 task_id 或其他唯一标识符（如果存在）
        task_id_slug = str(original_sample.get('task_id', f"sample_{sample_idx}")).replace('/', '_')
        
        filename = f"step_{step}_{task_id_slug}_{timestamp}.json"
        filepath = os.path.join(self.samples_dir, filename)
        
        # 从 original_sample 中提取原始问题描述
        orig_prompt_for_log = original_sample.get('original_prompt_for_debug', 'N/A')
        if orig_prompt_for_log == 'N/A':
            orig_prompt_for_log = original_sample.get('original_enhanced_prompt', 'N/A')

        sample_data_to_save = {
            "step": step,
            "timestamp_iso": datetime.now().isoformat(),
            "dataset_original_sample_info": {
                "level": original_sample.get('level'),
                "complexity_score": original_sample.get('complexity_score'),
                "original_problem_desc_for_debug": orig_prompt_for_log[:300], # 原始问题描述
                "testbench_path": original_sample.get('testbench_path', ''),
                "task_id": original_sample.get('task_id')
            },
            "model_input_prompt_preview": generated_result.get('model_input_prompt', '')[:300] + "...", # 模型实际收到的输入（预览）
            "generated_result": { # 只包含生成结果的核心部分
                'reasoning': generated_result.get('reasoning'),
                'code': generated_result.get('code'),
                'raw_output_preview': generated_result.get('raw_output', '')[:300] + "...", # LLM原始输出（预览）
                'generation_time_seconds': generated_result.get('generation_time')
            }
            # 为了避免文件过大，不直接保存完整的 prompt 和 raw_output，除非确实需要
            # 如果需要完整保存，取消下面两行的注释：
            # "full_model_input_prompt": generated_result.get('model_input_prompt'),
            # "full_raw_llm_output": generated_result.get('raw_output'),
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample_data_to_save, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved generation sample to {filepath}")
        except Exception as e_save:
            logger.error(f"Failed to save generation sample to {filepath}: {e_save}", exc_info=True)
    
    def _run_quick_test(self, code, sample):
        """快速功能测试"""
        try:
            if not code or not code.strip():
                return {"status": "empty_code"}
            
            # 简单的语法检查
            if 'module' not in code.lower():
                return {"status": "no_module"}
            
            if 'endmodule' not in code.lower():
                return {"status": "no_endmodule"}
            
            # 如果有testbench，可以尝试快速仿真
            testbench_path = sample.get('testbench_path', '')
            if os.path.exists(testbench_path):
                # 这里可以调用 run_iverilog_simulation 进行快速测试
                # 为了节省时间，可以设置较短的超时时间
                return {"status": "testbench_available", "path": testbench_path}
            
            return {"status": "basic_syntax_ok"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

def parse_llm_completion(completion_text: str, debug_prompt: Optional[str] = None, 
                        debug_context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses the LLM's completion text to extract reasoning and code blocks.
    Now supports both old format (VERILOG_REASONING/CODE markers) and new format (<think>/```verilog).
    Returns (reasoning_text, code_text).
    If a block is not found, the corresponding return value will be None.
    If no tags are found, assumes the entire output might be code (as a fallback).
    
    Args:
        completion_text: The LLM's output text to parse
        debug_prompt: Optional prompt that was used to generate this completion (for debugging)
        debug_context: Optional context dictionary with step, sample_idx, etc.
    """
    
    # === 调试信息初始化 ===
    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    debug_info = {
        "input_length": len(completion_text) if completion_text else 0,
        "has_debug_prompt": debug_prompt is not None,
        "debug_context": debug_context or {},
        "parsing_steps": [],
        "detected_patterns": [],
        "warnings": [],
        "errors": []
    }
    
    # === 输入验证 ===
    if not completion_text or not isinstance(completion_text, str):
        debug_info["errors"].append("Empty or invalid input")
        if debug_enabled:
            logger.debug("="*80)
            logger.debug("UTILS: parse_llm_completion: DEBUGGING SESSION START")
            logger.debug(f"Input validation FAILED: Empty or invalid input (type: {type(completion_text)})")
            if debug_prompt:
                logger.debug(f"Original Prompt (first 300 chars):\n{debug_prompt[:300]}{'...' if len(debug_prompt) > 300 else ''}")
            logger.debug("="*80)
        return None, None
    
    completion_text = completion_text.strip()
    debug_info["input_length_after_strip"] = len(completion_text)
    
    if len(completion_text) < 5:  # 太短的输入无意义
        debug_info["errors"].append(f"Input too short ({len(completion_text)} chars)")
        if debug_enabled:
            logger.debug("="*80)
            logger.debug("UTILS: parse_llm_completion: DEBUGGING SESSION START")
            logger.debug(f"Input validation FAILED: Too short ({len(completion_text)} chars)")
            if debug_prompt:
                logger.debug(f"Original Prompt (first 300 chars):\n{debug_prompt[:300]}{'...' if len(debug_prompt) > 300 else ''}")
            logger.debug(f"Short completion text: '{completion_text}'")
            logger.debug("="*80)
        return None, None
    
    # === 调试会话开始 ===
    if debug_enabled:
        logger.debug("="*80)
        logger.debug("UTILS: parse_llm_completion: DEBUGGING SESSION START")
        logger.debug(f"Debug Context: {debug_context}")
        logger.debug(f"Completion text length: {len(completion_text)} chars")
        
        # 打印原始prompt（如果提供）
        if debug_prompt:
            logger.debug("-" * 40)
            logger.debug("ORIGINAL PROMPT:")
            logger.debug("-" * 40)
            # 截断过长的prompt，但保留结构
            if len(debug_prompt) > 1000:
                prompt_preview = debug_prompt[:500] + "\n\n[... TRUNCATED ...]\n\n" + debug_prompt[-500:]
            else:
                prompt_preview = debug_prompt
            logger.debug(f"{prompt_preview}")
            logger.debug("-" * 40)
        
        # 打印completion文本预览
        logger.debug("COMPLETION TEXT PREVIEW:")
        logger.debug("-" * 40)
        if len(completion_text) > 800:
            completion_preview = completion_text[:400] + "\n\n[... MIDDLE TRUNCATED ...]\n\n" + completion_text[-400:]
        else:
            completion_preview = completion_text
        logger.debug(f"{completion_preview}")
        logger.debug("-" * 40)
    
    reasoning_part = None
    code_part = None
    detected_format = "no_known_format"
    parsing_success_flags = {
        "think_block": False,
        "verilog_block": False,
        "generic_code_block": False,
        "module_pattern": False,
        "emergency_extraction": False,
        "full_output_fallback": False
    }

    try:
        debug_info["parsing_steps"].append("Starting main parsing logic")
        
        # === 新格式解析：<think>...</think> ===
        think_match = re.search(
            r"<think>(.*?)</think>",
            completion_text,
            re.DOTALL | re.IGNORECASE
        )
        if think_match:
            reasoning_part = think_match.group(1).strip()
            parsing_success_flags["think_block"] = True
            debug_info["detected_patterns"].append("think_block")
            debug_info["parsing_steps"].append(f"Found <think> block with {len(reasoning_part)} chars")
            if debug_enabled:
                logger.debug(f"✓ FOUND <think> block:")
                logger.debug(f"  Length: {len(reasoning_part)} chars")
                logger.debug(f"  Preview: '{reasoning_part[:150]}{'...' if len(reasoning_part) > 150 else ''}'")
        else:
            debug_info["warnings"].append("<think> block not found")
            debug_info["parsing_steps"].append("No <think> block found")
            if debug_enabled:
                logger.debug("✗ <think> block NOT FOUND")
                # 检查是否有不完整的think标签
                if "<think>" in completion_text.lower():
                    logger.debug("  ⚠️ Found opening <think> tag but no closing </think>")
                elif "</think>" in completion_text.lower():
                    logger.debug("  ⚠️ Found closing </think> tag but no opening <think>")

        # === 新格式解析：```verilog...``` ===
        verilog_code_match = re.search(
            r"```verilog\s*(.*?)\s*```",
            completion_text,
            re.DOTALL | re.IGNORECASE
        )
        if verilog_code_match:
            code_part = verilog_code_match.group(1).strip()
            parsing_success_flags["verilog_block"] = True
            debug_info["detected_patterns"].append("verilog_block")
            debug_info["parsing_steps"].append(f"Found ```verilog block with {len(code_part)} chars")
            if debug_enabled:
                logger.debug(f"✓ FOUND ```verilog block:")
                logger.debug(f"  Length: {len(code_part)} chars")
                logger.debug(f"  Preview: '{code_part[:150]}{'...' if len(code_part) > 150 else ''}'")
        else:
            debug_info["warnings"].append("```verilog block not found")
            debug_info["parsing_steps"].append("No ```verilog block found, trying fallbacks")
            if debug_enabled:
                logger.debug("✗ ```verilog block NOT FOUND")
                # 检查是否有不完整的代码块
                if "```verilog" in completion_text.lower():
                    logger.debug("  ⚠️ Found opening ```verilog but no closing ```")
                elif "```" in completion_text:
                    logger.debug("  ⚠️ Found generic ``` markers but not verilog-specific")
            
            # Fallback: 尝试通用的代码块格式 ```...```
            generic_code_match = re.search(
                r"```\s*(module\s+.*?endmodule)\s*```",
                completion_text,
                re.DOTALL | re.IGNORECASE
            )
            if generic_code_match:
                code_part = generic_code_match.group(1).strip()
                parsing_success_flags["generic_code_block"] = True
                debug_info["detected_patterns"].append("generic_code_block")
                debug_info["parsing_steps"].append(f"Found generic code block (fallback) with {len(code_part)} chars")
                detected_format = "fallback_generic_code_block"
                if debug_enabled:
                    logger.debug(f"✓ FOUND generic ``` block (fallback):")
                    logger.debug(f"  Length: {len(code_part)} chars")
                    logger.debug(f"  Preview: '{code_part[:150]}{'...' if len(code_part) > 150 else ''}'")

        # === 格式检测和日志 ===
        if think_match and verilog_code_match:
            detected_format = "new_format_think_and_code"
        elif think_match and not verilog_code_match:
            detected_format = "new_format_think_only"
        elif not think_match and verilog_code_match:
            detected_format = "new_format_code_only"

        # === 详细的格式缺失警告 ===
        if not think_match or not verilog_code_match:
            missing_parts = []
            if not think_match:
                missing_parts.append("<think>...</think>")
            if not verilog_code_match:
                missing_parts.append("```verilog...```")
            
            warning_msg = f"Missing format blocks: {', '.join(missing_parts)}"
            debug_info["warnings"].append(warning_msg)
            
            if debug_enabled:
                logger.debug(f"⚠️ FORMAT WARNING: {warning_msg}")
                # 显示完整的completion用于分析
                logger.debug("FULL COMPLETION TEXT FOR ANALYSIS:")
                logger.debug("-" * 40)
                logger.debug(completion_text)
                logger.debug("-" * 40)

        # === 额外的代码提取fallback ===
        if code_part is None:
            debug_info["parsing_steps"].append("Attempting direct module...endmodule fallback")
            if debug_enabled:
                logger.debug("🔄 Attempting direct module...endmodule fallback")
            
            module_match = re.search(
                r"(module\s+\w+.*?endmodule)",
                completion_text,
                re.DOTALL | re.IGNORECASE
            )
            if module_match:
                code_part = module_match.group(1).strip()
                parsing_success_flags["module_pattern"] = True
                debug_info["detected_patterns"].append("module_pattern")
                debug_info["parsing_steps"].append(f"Found module pattern (fallback) with {len(code_part)} chars")
                if detected_format in ["no_known_format", "new_format_think_only"]:
                    detected_format = "fallback_module_endmodule"
                if debug_enabled:
                    logger.debug(f"✓ FOUND module pattern (fallback):")
                    logger.debug(f"  Length: {len(code_part)} chars")
                    logger.debug(f"  Preview: '{code_part[:150]}{'...' if len(code_part) > 150 else ''}'")
            else:
                debug_info["warnings"].append("Direct module...endmodule pattern not found")
                if debug_enabled:
                    logger.debug("✗ Direct module...endmodule pattern NOT FOUND")

        # === Final fallback ===
        if code_part is None and reasoning_part is None:
            debug_info["parsing_steps"].append("Attempting final fallback strategies")
            if debug_enabled:
                logger.debug("🔄 Attempting final fallback strategies")
            
            has_markers = any(marker in completion_text for marker in [
                "<think>", "</think>", "```verilog", "```"
            ])
            debug_info["has_format_markers"] = has_markers
            
            if not has_markers:
                debug_info["parsing_steps"].append("No format markers found, treating full output as code")
                code_part = completion_text.strip()
                parsing_success_flags["full_output_fallback"] = True
                if code_part:
                    detected_format = "fallback_full_output_as_code"
                if debug_enabled:
                    logger.debug("🔄 No format markers found, treating full output as code")
                    logger.debug(f"  Full output length: {len(code_part)} chars")
            else:
                debug_info["warnings"].append("Format markers present but parsing failed")
                if debug_enabled:
                    logger.debug("⚠️ Format markers were present but parsing failed")
                
                # 尝试恢复不完整的think标签
                if "</think>" in completion_text and "<think>" not in completion_text:
                    think_end_pos = completion_text.find("</think>")
                    potential_reasoning = completion_text[:think_end_pos].strip()
                    if len(potential_reasoning) > 10:
                        reasoning_part = potential_reasoning
                        debug_info["parsing_steps"].append("Recovered reasoning from incomplete think tags")
                        if detected_format == "no_known_format":
                            detected_format = "fallback_recovered_think_only"
                        if debug_enabled:
                            logger.debug(f"✓ RECOVERED reasoning from incomplete tags:")
                            logger.debug(f"  Length: {len(reasoning_part)} chars")

    except Exception as e:
        error_msg = f"Exception during parsing: {e}"
        debug_info["errors"].append(error_msg)
        debug_info["parsing_steps"].append(f"Exception occurred: {str(e)}")
        if debug_enabled:
            logger.debug(f"❌ EXCEPTION during parsing: {e}")
        
        logger.error(f"UTILS: parse_llm_completion: Exception during parsing: {e}", exc_info=True)
        
        # Emergency extraction
        if "module" in completion_text.lower() and "endmodule" in completion_text.lower():
            try:
                debug_info["parsing_steps"].append("Attempting emergency extraction")
                module_match = re.search(r"(module\s+.*?endmodule)", completion_text, re.DOTALL | re.IGNORECASE)
                if module_match:
                    code_part = module_match.group(1).strip()
                    parsing_success_flags["emergency_extraction"] = True
                    debug_info["detected_patterns"].append("emergency_extraction")
                    if detected_format == "no_known_format":
                        detected_format = "fallback_emergency_extraction"
                    if debug_enabled:
                        logger.debug(f"✓ EMERGENCY extraction successful:")
                        logger.debug(f"  Length: {len(code_part)} chars")
            except Exception as e_emergency:
                emergency_error = f"Exception during emergency extraction: {e_emergency}"
                debug_info["errors"].append(emergency_error)
                if debug_enabled:
                    logger.debug(f"❌ Emergency extraction failed: {e_emergency}")
                logger.error(f"UTILS: parse_llm_completion: {emergency_error}", exc_info=True)

    # === 数据质量检查和清理 ===
    original_reasoning_length = len(reasoning_part) if reasoning_part else 0
    original_code_length = len(code_part) if code_part else 0
    
    if reasoning_part:
        # 清理推理部分
        reasoning_part = re.sub(r"^[\s\n]*", "", reasoning_part)
        reasoning_part = re.sub(r"[\s\n]*$", "", reasoning_part)
        reasoning_part = re.sub(r"^(your thought process|思考过程).*?[:：]\s*", "", reasoning_part, flags=re.IGNORECASE)
        
        if len(reasoning_part) < 10:
            debug_info["warnings"].append(f"Reasoning too short ({len(reasoning_part)} chars), setting to None")
            reasoning_part = None

    if code_part:
        # 清理代码部分
        code_part = re.sub(r"^[\s\n]*", "", code_part)
        code_part = re.sub(r"[\s\n]*$", "", code_part)
        
        if "module" not in code_part.lower():
            debug_info["warnings"].append(f"Extracted code doesn't contain 'module' keyword")
        
        if len(code_part) < 20:
            debug_info["warnings"].append(f"Code too short ({len(code_part)} chars), setting to None")
            code_part = None

    # === 最终调试总结 ===
    debug_info["final_results"] = {
        "reasoning_length": len(reasoning_part) if reasoning_part else 0,
        "code_length": len(code_part) if code_part else 0,
        "detected_format": detected_format,
        "parsing_success_flags": parsing_success_flags,
        "original_reasoning_length": original_reasoning_length,
        "original_code_length": original_code_length
    }
    
    if debug_enabled:
        logger.debug("="*40)
        logger.debug("PARSING SUMMARY:")
        logger.debug(f"  Detected format: {detected_format}")
        logger.debug(f"  Success flags: {parsing_success_flags}")
        logger.debug(f"  Final reasoning: {len(reasoning_part) if reasoning_part else 0} chars")
        logger.debug(f"  Final code: {len(code_part) if code_part else 0} chars")
        logger.debug(f"  Warnings: {len(debug_info['warnings'])}")
        logger.debug(f"  Errors: {len(debug_info['errors'])}")
        
        if debug_info["warnings"]:
            logger.debug("  Warning details:")
            for i, warning in enumerate(debug_info["warnings"], 1):
                logger.debug(f"    {i}. {warning}")
        
        if debug_info["errors"]:
            logger.debug("  Error details:")
            for i, error in enumerate(debug_info["errors"], 1):
                logger.debug(f"    {i}. {error}")
        
        # 如果解析失败，显示额外诊断信息
        if not reasoning_part and not code_part:
            logger.debug("❌ PARSING COMPLETELY FAILED")
            logger.debug("Diagnostic information:")
            logger.debug(f"  Contains '<think>': {'<think>' in completion_text.lower()}")
            logger.debug(f"  Contains '</think>': {'</think>' in completion_text.lower()}")
            logger.debug(f"  Contains '```verilog': {'```verilog' in completion_text.lower()}")
            logger.debug(f"  Contains '```': {'```' in completion_text}")
            logger.debug(f"  Contains 'module': {'module' in completion_text.lower()}")
            logger.debug(f"  Contains 'endmodule': {'endmodule' in completion_text.lower()}")
        elif reasoning_part and code_part:
            logger.debug("✅ PARSING SUCCESSFUL (both parts found)")
        else:
            logger.debug("⚠️ PARTIAL PARSING SUCCESS")
        
        logger.debug("="*80)
        logger.debug("UTILS: parse_llm_completion: DEBUGGING SESSION END")
        logger.debug("="*80)

    return reasoning_part, code_part


# 辅助函数：在调用解析函数时传递调试信息
def parse_llm_completion_with_context(completion_text: str, prompt: str = None, 
                                    step: int = None, sample_idx: int = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Wrapper function that calls parse_llm_completion with debug context
    """
    debug_context = {}
    if step is not None:
        debug_context["step"] = step
    if sample_idx is not None:
        debug_context["sample_idx"] = sample_idx
    
    return parse_llm_completion(completion_text, debug_prompt=prompt, debug_context=debug_context)

def validate_and_fix_output_format(raw_output: str) -> Tuple[str, bool]:
    """
    验证和修复LLM输出格式
    返回: (修复后的输出, 是否需要修复)
    """
    if not raw_output:
        return raw_output, False
    
    original_output = raw_output
    needs_fix = False
    
    # 移除Qwen对话标记
    if '<|im_start|>' in raw_output or '<|im_end|>' in raw_output:
        raw_output = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', raw_output, flags=re.DOTALL)
        raw_output = re.sub(r'<\|im_start\|>assistant\n?', '', raw_output)
        raw_output = re.sub(r'<\|im_end\|>', '', raw_output)
        needs_fix = True
    
    # 检查是否有think标签
    has_think_start = '<think>' in raw_output.lower()
    has_think_end = '</think>' in raw_output.lower()
    
    # 检查是否有代码块
    has_verilog_block = '```verilog' in raw_output.lower()
    has_code_end = '```' in raw_output and raw_output.rfind('```') > raw_output.find('```verilog')
    
    # 如果缺少think标签，尝试添加
    if not has_think_start and not has_think_end:
        # 查找可能的思考内容（在代码块之前的文本）
        verilog_pos = raw_output.lower().find('```verilog')
        if verilog_pos > 0:
            thinking_content = raw_output[:verilog_pos].strip()
            if len(thinking_content) > 50:  # 有足够的内容
                raw_output = f"<think>\n{thinking_content}\n</think>\n\n{raw_output[verilog_pos:]}"
                needs_fix = True
    
    # 如果只有开始标签，尝试修复结束标签
    elif has_think_start and not has_think_end:
        verilog_pos = raw_output.lower().find('```verilog')
        if verilog_pos > 0:
            think_pos = raw_output.lower().find('<think>')
            think_content = raw_output[think_pos:verilog_pos].strip()
            remaining_content = raw_output[verilog_pos:]
            raw_output = f"{think_content}\n</think>\n\n{remaining_content}"
            needs_fix = True
    
    # 如果缺少verilog代码块标记，尝试添加
    if not has_verilog_block:
        # 查找module...endmodule模式
        module_match = re.search(r'(module\s+.*?endmodule)', raw_output, re.DOTALL | re.IGNORECASE)
        if module_match:
            module_code = module_match.group(1)
            # 替换原始的module代码为带标记的版本
            raw_output = raw_output.replace(module_code, f"```verilog\n{module_code}\n```")
            needs_fix = True
    
    # 如果有开始标记但没有结束标记
    elif has_verilog_block and not has_code_end:
        verilog_pos = raw_output.lower().find('```verilog')
        if verilog_pos >= 0:
            # 查找verilog代码块后的endmodule
            after_verilog = raw_output[verilog_pos + 10:]  # 跳过```verilog
            endmodule_match = re.search(r'(.*?endmodule)', after_verilog, re.DOTALL | re.IGNORECASE)
            if endmodule_match:
                verilog_content = endmodule_match.group(1).strip()
                remaining_content = after_verilog[endmodule_match.end():].strip()
                
                # 重构输出
                before_verilog = raw_output[:verilog_pos]
                if remaining_content:
                    raw_output = f"{before_verilog}```verilog\n{verilog_content}\n```\n{remaining_content}"
                else:
                    raw_output = f"{before_verilog}```verilog\n{verilog_content}\n```"
                needs_fix = True
    
    return raw_output.strip(), needs_fix

class FormatValidationCallback(TrainerCallback):
    """格式验证回调，用于监控和修复输出格式问题"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.format_issues_count = 0
        self.format_fixes_count = 0
        
        # 创建格式问题日志目录
        self.format_log_dir = os.path.join(output_dir, "format_issues")
        os.makedirs(self.format_log_dir, exist_ok=True)
    
    def validate_generation_format(self, raw_output: str, step: int, sample_idx: int) -> Dict[str, Any]:
        """验证单个生成样本的格式"""
        fixed_output, needs_fix = validate_and_fix_output_format(raw_output)
        
        # 解析修复后的输出
        reasoning, code = parse_llm_completion(fixed_output)
        
        validation_result = {
            "original_output": raw_output,
            "fixed_output": fixed_output if needs_fix else None,
            "needs_fix": needs_fix,
            "has_reasoning": reasoning is not None,
            "has_code": code is not None,
            "is_valid_format": reasoning is not None and code is not None,
            "step": step,
            "sample_idx": sample_idx
        }
        
        # 记录格式问题
        if needs_fix:
            self.format_fixes_count += 1
            self._log_format_issue(validation_result)
        
        if not validation_result["is_valid_format"]:
            self.format_issues_count += 1
            self._log_format_issue(validation_result, severe=True)
        
        return validation_result
    
    def _log_format_issue(self, validation_result: Dict[str, Any], severe: bool = False):
        """记录格式问题到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        issue_type = "SEVERE" if severe else "FIXED"
        filename = f"format_issue_{issue_type}_{validation_result['step']}_{validation_result['sample_idx']}_{timestamp}.json"
        filepath = os.path.join(self.format_log_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(validation_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to log format issue: {e}")
    
    def get_format_stats(self) -> Dict[str, int]:
        """获取格式问题统计"""
        return {
            "format_issues_count": self.format_issues_count,
            "format_fixes_count": self.format_fixes_count
        }

def extract_module_info(verilog_file: str) -> Tuple[str, List[str]]:
    """ 
    Extracts module name and port list from a Verilog file.
    Updated to handle the new folder-prefixed paths.
    """
    try:
        # 检查文件是否存在（处理可能的路径问题）
        if not os.path.exists(verilog_file):
            # 尝试相对路径
            alt_path = os.path.join(".", verilog_file)
            if os.path.exists(alt_path):
                verilog_file = alt_path
            else:
                logger.error(f"UTILS: Verilog file not found: {verilog_file}")
                return "", []
        
        with open(verilog_file, "r", encoding="utf-8") as f:
            content = f.read()

        module_match = re.search(r"\bmodule\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", content, re.IGNORECASE)
        if not module_match:
            logger.error(f"UTILS: No module declaration found in {verilog_file}")
            return "", []
        module_name = module_match.group(1)

        port_pattern_text = r"module\s+" + re.escape(module_name) + r"\s*(?:#\s*\(.*?\)\s*)?\((.*?)\)\s*;"
        port_match = re.search(port_pattern_text, content, re.IGNORECASE | re.DOTALL)

        ports = []
        if port_match:
            port_text = port_match.group(1)
            port_text = re.sub(r"//.*?(\n|$)", "\n", port_text) 
            port_text = re.sub(r"/\*.*?\*/", "", port_text, flags=re.DOTALL)
            port_text = port_text.replace("\n", " ").strip()

            if port_text:
                port_declarations = [p.strip() for p in port_text.split(',') if p.strip()]
                for port_decl_full in port_declarations:
                    parts = port_decl_full.split()
                    if parts:
                        potential_name = parts[-1].strip("(),;")
                        verilog_keywords = {
                            "input", "output", "inout", "reg", "wire", "logic", "signed", "unsigned",
                            "parameter", "localparam", "integer", "real", "time", "genvar",
                            "always", "assign", "begin", "end", "if", "else", "case", "for"
                        }
                        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", potential_name) and potential_name.lower() not in verilog_keywords:
                            ports.append(potential_name)
                        elif len(parts) > 1 and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", parts[-2].strip("(),;")) and parts[-2].lower() not in verilog_keywords:
                            ports.append(parts[-2].strip("(),;"))

        unique_ports = sorted(list(set(ports)))
        logger.debug(f"UTILS: Extracted from {verilog_file}: module='{module_name}', ports={unique_ports}")
        return module_name, unique_ports

    except FileNotFoundError:
        logger.error(f"UTILS: Verilog file not found: {verilog_file}")
        return "", []
    except Exception as e:
        logger.error(f"UTILS: Error reading or parsing Verilog file {verilog_file}: {e}", exc_info=True)
        return "", []


def assess_code_quality(verilog_code: str) -> Dict[str, float]:
    """
    Assess various quality metrics of Verilog code.
    Returns a dictionary with quality scores (0-1 range).
    """
    if not verilog_code or not verilog_code.strip():
        return {"complexity": 0, "readability": 0, "efficiency": 0, "structure": 0}
    
    lines = verilog_code.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # Calculate complexity score (inverse of actual complexity)
    nested_depth = 0
    max_nested_depth = 0
    current_depth = 0
    
    always_blocks = 0
    assign_statements = 0
    case_statements = 0
    if_statements = 0
    
    for line in non_empty_lines:
        line_lower = line.lower()
        
        # Count nesting depth
        if any(keyword in line_lower for keyword in ['begin', 'case', 'if']):
            current_depth += 1
            max_nested_depth = max(max_nested_depth, current_depth)
        if any(keyword in line_lower for keyword in ['end', 'endcase']):
            current_depth = max(0, current_depth - 1)
        
        # Count different constructs
        if 'always' in line_lower:
            always_blocks += 1
        if 'assign' in line_lower:
            assign_statements += 1
        if 'case' in line_lower:
            case_statements += 1
        if 'if' in line_lower and 'ifdef' not in line_lower:
            if_statements += 1
    
    # Calculate scores
    complexity_score = max(0, 1.0 - (max_nested_depth / 10.0))
    
    # Readability: comments, spacing, naming conventions
    comment_lines = sum(1 for line in lines if '//' in line or '/*' in line)
    comment_ratio = comment_lines / max(1, len(non_empty_lines))
    readability_score = min(1.0, comment_ratio * 2 + 0.3)
    
    # Efficiency: prefer assign over always for combinational logic
    total_logic = always_blocks + assign_statements
    efficiency_score = 0.7 if total_logic > 0 else 0.0
    if assign_statements > always_blocks * 2:
        efficiency_score += 0.3
    
    # Structure: balanced use of different constructs
    construct_diversity = len([x for x in [always_blocks, assign_statements, case_statements] if x > 0])
    structure_score = min(1.0, construct_diversity / 3.0 + 0.4)
    
    return {
        "complexity": complexity_score,
        "readability": readability_score,
        "efficiency": efficiency_score,
        "structure": structure_score
    }


def assess_design_complexity(verilog_file: str) -> float:
    """
    Assess the complexity level of a reference design for curriculum learning.
    Returns complexity score (0-10).
    Updated to handle new paths.
    """
    try:
        # 处理可能的路径问题
        if not os.path.exists(verilog_file):
            alt_path = os.path.join(".", verilog_file)
            if os.path.exists(alt_path):
                verilog_file = alt_path
            else:
                logger.warning(f"Could not find Verilog file for complexity assessment: {verilog_file}")
                return 5.0  # 默认中等复杂度
        
        with open(verilog_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Count complexity indicators
        always_blocks = len(re.findall(r'\balways\b', content, re.IGNORECASE))
        state_machines = len(re.findall(r'\bcase\b.*?\bendcase\b', content, re.IGNORECASE | re.DOTALL))
        sequential_logic = len(re.findall(r'@\s*\(\s*posedge|@\s*\(\s*negedge', content, re.IGNORECASE))
        memory_elements = len(re.findall(r'\breg\b|\bmemory\b', content, re.IGNORECASE))
        
        # Calculate complexity score
        complexity = (
            len(non_empty_lines) / 10 +
            always_blocks * 0.5 +
            state_machines * 2.0 +
            sequential_logic * 1.5 +
            memory_elements * 1.0
        )
        
        return min(10.0, complexity)
        
    except Exception as e:
        logger.warning(f"Could not assess complexity for {verilog_file}: {e}")
        return 5.0


def parse_simulation_results_from_output(sim_output: str) -> Tuple[int, int, int, bool]:
    """
    Parses simulation output for passed/failed test cases and overall status.
    """
    passed_count, failed_count, total_cases = 0, 0, 0
    is_overall_pass = False

    summary_match = re.search(
        r"Passed:\s*(\d+)\s*,\s*Failed:\s*(\d+).*?"
        r"Total Test Cases:\s*(\d+).*?"
        r"(OVERALL_PASS|OVERALL_FAIL)",
        sim_output,
        re.DOTALL | re.IGNORECASE,
    )

    if summary_match:
        try:
            passed_count = int(summary_match.group(1))
            failed_count = int(summary_match.group(2))
            total_cases = int(summary_match.group(3))
            is_overall_pass = summary_match.group(4).upper() == "OVERALL_PASS"
        except ValueError:
            logger.warning("UTILS: ValueError parsing full summary. Fallback if OVERALL_PASS found.")
            if re.search(r"OVERALL_PASS", sim_output, re.IGNORECASE):
                is_overall_pass = True
    else:
        pc_match = re.search(r"Passed:\s*(\d+)", sim_output, re.IGNORECASE)
        fc_match = re.search(r"Failed:\s*(\d+)", sim_output, re.IGNORECASE)
        tc_match = re.search(r"Total Test Cases:\s*(\d+)", sim_output, re.IGNORECASE)

        if pc_match: passed_count = int(pc_match.group(1))
        if fc_match: failed_count = int(fc_match.group(1))
        if tc_match: total_cases = int(tc_match.group(1))

        if total_cases == 0 and (passed_count > 0 or failed_count > 0):
            total_cases = passed_count + failed_count
        
        if re.search(r"OVERALL_PASS", sim_output, re.IGNORECASE):
            is_overall_pass = True
            if total_cases == 0 and passed_count > 0 and failed_count == 0:
                total_cases = passed_count
        elif passed_count > 0 and failed_count == 0 and not re.search(r"OVERALL_FAIL", sim_output, re.IGNORECASE):
            is_overall_pass = True
            if total_cases == 0: total_cases = passed_count
    
    if total_cases == 0 and (passed_count > 0 or failed_count > 0):
        total_cases = passed_count + failed_count
            
    return passed_count, failed_count, total_cases, is_overall_pass


def validate_verilog_code(
    completion_code: str, module_name: str, required_ports: List[str]
) -> Tuple[bool, str]:
    """
    Validates basic structure of generated Verilog code.
    """
    if not module_name:
        return False, "Validation Error: No valid module name provided."
    if not completion_code or not completion_code.strip():
        return False, "Validation Error: Generated code is empty."
    
    clean_code = completion_code.strip()

    if not re.match(r"^\s*module\s+", clean_code, re.IGNORECASE):
        return False, "Validation Error: Does not start with 'module'."
    
    generated_module_match = re.search(r"^\s*module\s+([\w\.]+)\s*(?:#\(.*?\))?\s*\(", clean_code, re.IGNORECASE | re.MULTILINE)
    if not generated_module_match:
        return False, f"Validation Error: Cannot find module declaration for '{module_name}'."
    
    declared_module_name = generated_module_match.group(1)
    if declared_module_name != module_name:
        return False, f"Validation Error: Declared module name '{declared_module_name}' != expected '{module_name}'."

    if not re.search(r"\bendmodule\b", clean_code, re.IGNORECASE | re.MULTILINE):
        return False, "Validation Error: Missing 'endmodule'."

    if required_ports:
        port_pattern_generated = rf"^\s*module\s+{re.escape(module_name)}\s*(?:#\(.*?\)\s*)?\((.*?)\)\s*;"
        port_match_generated = re.search(port_pattern_generated, clean_code, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if not port_match_generated:
            return False, f"Validation Error: Module '{module_name}' port list not found/malformed."
        
        port_text_generated = port_match_generated.group(1)
        port_text_generated = re.sub(r"//.*?(\n|$)", "\n", port_text_generated)
        port_text_generated = re.sub(r"/\*.*?\*/", "", port_text_generated, flags=re.DOTALL)
        port_text_generated = port_text_generated.replace("\n", " ").strip()

        missing_ports = [rp for rp in required_ports if not re.search(rf"\b{re.escape(rp)}\b", port_text_generated)]
        if missing_ports:
            return False, f"Validation Error: Missing ports in '{module_name}': {', '.join(missing_ports)}. Declared: '{port_text_generated[:200]}...'"

    return True, ""


def run_iverilog_simulation(
    generated_verilog_code: str,
    testbench_file_path: str,
    expected_total_tests_from_manifest: int,
    prompt_identifier: str = "N/A",
    completion_idx: int = -1,
    print_simulation_details: bool = False 
) -> Dict[str, Any]:
    """
    Runs Icarus Verilog simulation with enhanced error handling.
    Updated to handle new path format.
    """
    result = {
        "compilation_success": False, "simulation_run_success": False, "parsing_success": False,
        "passed_tests": 0, "failed_tests": 0, "total_tests_in_output": 0,
        "all_tests_passed_by_tb": False, "error_message": None, "raw_stdout": "", "raw_stderr": "",
        "quality_metrics": {}
    }

    log_sim_header = f"--- SIMULATION RUN FOR PROMPT='{prompt_identifier[:60]}...', COMPLETION_IDX={completion_idx} ---"
    if print_simulation_details:
        logger.debug("\n" + "="*80 + f"\n{log_sim_header}")

    # 处理可能的路径问题
    if not os.path.exists(testbench_file_path):
        # 尝试相对路径
        alt_path = os.path.join(".", testbench_file_path)
        if os.path.exists(alt_path):
            testbench_file_path = alt_path
        else:
            result["error_message"] = f"Sim Error: Testbench not found: {testbench_file_path}."
            logger.error(f"UTILS: {result['error_message']}")
            return result

    if not generated_verilog_code or not generated_verilog_code.strip():
        result["error_message"] = "Sim Error: Empty Verilog code."
        logger.info(f"UTILS: {result['error_message']}")
        return result
    
    # Assess code quality before simulation
    result["quality_metrics"] = assess_code_quality(generated_verilog_code)
    
    try:
        with open(testbench_file_path, "r", encoding="utf-8") as tb_f:
            testbench_content = tb_f.read()
    except Exception as e:
        result["error_message"] = f"Sim Error: Could not read testbench {testbench_file_path}: {e}"
        logger.error(f"UTILS: {result['error_message']}")
        return result

    if print_simulation_details:
        logger.debug(f"\n[Testbench Path]: {testbench_file_path}")
        logger.debug(f"[Generated Verilog Code (DUT)]:\n{generated_verilog_code}\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        if print_simulation_details: 
            logger.debug(f"[Temp Sim Dir]: {temp_dir}")
        generated_design_file = os.path.join(temp_dir, "design_under_test.v")
        compiled_output_file = os.path.join(temp_dir, "simulation.vvp")

        with open(generated_design_file, "w", encoding="utf-8") as f: 
            f.write(generated_verilog_code)

        tb_top_module_name = "testbench"
        match = re.search(r"^\s*module\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(\s*\))?\s*;", testbench_content, re.MULTILINE | re.IGNORECASE)
        if match: 
            tb_top_module_name = match.group(1)
        if print_simulation_details: 
            logger.debug(f"[Deduced TB Top Module]: {tb_top_module_name}")
        
        compile_command = ["iverilog", "-g2012", "-Wall", "-o", compiled_output_file, generated_design_file, testbench_file_path, "-s", tb_top_module_name]
        if print_simulation_details: 
            logger.debug(f"\n[Compile Command]: {' '.join(compile_command)}")

        try:
            process_compile = subprocess.run(compile_command, capture_output=True, text=True, timeout=15, errors="replace")
            result["raw_stdout"] += f"IVERILOG COMPILE STDOUT:\n{process_compile.stdout}\n"
            result["raw_stderr"] += f"IVERILOG COMPILE STDERR:\n{process_compile.stderr}\n"
            
            if print_simulation_details:
                logger.debug("\n[Compiler Output]:")
                if process_compile.stdout: logger.debug(f"  STDOUT:\n{process_compile.stdout}")
                if process_compile.stderr: logger.debug(f"  STDERR:\n{process_compile.stderr}")
                logger.debug(f"  Return Code: {process_compile.returncode}")

            if process_compile.returncode != 0:
                result["error_message"] = f"Icarus Verilog compilation failed. Exit: {process_compile.returncode}. Stderr: {process_compile.stderr[:1000]}"
                if print_simulation_details: logger.debug(f"COMPILATION FAILED. Error: {result['error_message']}")
                return result
            result["compilation_success"] = True
            if print_simulation_details: logger.debug("COMPILATION SUCCEEDED.")
            
        except subprocess.TimeoutExpired:
            result["error_message"] = "Icarus Verilog compilation timed out."
            if print_simulation_details: logger.debug(f"COMPILATION TIMED OUT. Error: {result['error_message']}")
            return result
        except Exception as e:
            result["error_message"] = f"Error during iverilog compilation: {str(e)}"
            if print_simulation_details: logger.debug(f"COMPILATION EXCEPTION. Error: {result['error_message']}")
            return result

        execute_command = ["vvp", compiled_output_file]
        if print_simulation_details: logger.debug(f"\n[Execution Command]: {' '.join(execute_command)}")

        try:
            process_sim = subprocess.run(execute_command, capture_output=True, text=True, timeout=15, errors="replace")
            result["raw_stdout"] += f"\nVVP SIMULATION STDOUT:\n{process_sim.stdout}\n"
            result["raw_stderr"] += f"VVP SIMULATION STDERR:\n{process_sim.stderr}\n"

            if print_simulation_details:
                logger.debug("\n[Simulator Output (VVP)]:")
                if process_sim.stdout: logger.debug(f"  STDOUT:\n{process_sim.stdout}")
                if process_sim.stderr: logger.debug(f"  STDERR:\n{process_sim.stderr}")
                logger.debug(f"  Return Code: {process_sim.returncode}")

            if process_sim.returncode == 0 or "$finish" in process_sim.stdout.lower() or "vvp finished" in process_sim.stdout.lower():
                result["simulation_run_success"] = True
            elif "error" in process_sim.stderr.lower() or "segmentation fault" in process_sim.stderr.lower():
                result["simulation_run_success"] = False
                vvp_err_msg = f"\nVVP sim error. Exit: {process_sim.returncode}. Stdout: ...{process_sim.stdout[-500:]}. Stderr: {process_sim.stderr[:1000]}"
                current_err = result["error_message"]
                if current_err:
                    result["error_message"] = f"{str(current_err).strip()}\n{vvp_err_msg.strip()}"
                else:
                    result["error_message"] = vvp_err_msg.strip()
            else: 
                result["simulation_run_success"] = True 
                logger.warning(f"UTILS: VVP sim exit {process_sim.returncode}, but no explicit error. Proceeding. Stdout: {process_sim.stdout[:200]}")
            
            if print_simulation_details: 
                logger.debug(f"SIMULATION RUN CONSIDERED {'SUCCESSFUL' if result['simulation_run_success'] else 'FAILED'}.")

            if result["compilation_success"] and result["simulation_run_success"]:
                p, f, t, overall_pass = parse_simulation_results_from_output(process_sim.stdout)
                result.update({"passed_tests": p, "failed_tests": f, "total_tests_in_output": t, "all_tests_passed_by_tb": overall_pass})
                if print_simulation_details:
                    logger.debug("\n[Parsed Simulation Results]:")
                    logger.debug(f"  Passed: {p}, Failed: {f}, Total: {t}, Overall Pass: {overall_pass}")

                if t > 0 or p > 0 or f > 0 or overall_pass:
                    result["parsing_success"] = True
                elif result["simulation_run_success"]:
                    result["parsing_success"] = False
                    err_msg_parse = f"Could not parse VVP output for counts, or 0 tests. Raw VVP stdout: {process_sim.stdout[:500]}"
                    result["error_message"] = (result.get("error_message", "") + "\n" + err_msg_parse).strip()
                    logger.warning(f"UTILS: {err_msg_parse}")
                
                if result["parsing_success"] and expected_total_tests_from_manifest > 0 and t != expected_total_tests_from_manifest:
                    mismatch_msg = (f"Sim Mismatch! Parsed total ({t}) != expected ({expected_total_tests_from_manifest}) for TB: {testbench_file_path}.")
                    logger.warning(f"UTILS: {mismatch_msg} Output: {process_sim.stdout[:200]}")
                    
        except subprocess.TimeoutExpired:
            result["simulation_run_success"] = False
            current_error_message = result["error_message"]
            timeout_specific_message = "VVP simulation timed out."

            if current_error_message is None:
                result["error_message"] = timeout_specific_message
            else:
                result["error_message"] = f"{str(current_error_message).strip()}\n{timeout_specific_message}"
            
            if print_simulation_details:
                logger.debug(f"SIMULATION EXECUTION TIMED OUT. Error: {result['error_message']}")

        except Exception as e:
            result["simulation_run_success"] = False
            current_error_message = result["error_message"]
            exception_specific_message = f"Error during VVP execution or parsing: {str(e)}"

            if current_error_message is None:
                result["error_message"] = exception_specific_message
            else:
                result["error_message"] = f"{str(current_error_message).strip()}\n{exception_specific_message}"

            if print_simulation_details:
                logger.debug(f"SIMULATION EXECUTION EXCEPTION. Error: {result['error_message']}")
                
        if print_simulation_details and result.get("error_message"):
            logger.debug(f"SIMULATION ERROR/TIMEOUT. Error: {result['error_message']}")
            
    if print_simulation_details:
        logger.debug(f"--- END OF SIMULATION RUN FOR PROMPT='{prompt_identifier[:60]}...', COMPLETION_IDX={completion_idx} ---\n" + "="*80 + "\n")
    return result


def validate_and_update_dataset_paths(dataset: Dataset, dataset_base_path: str = None) -> List[Dict[str, Any]]:
    """
    Validates dataset examples, resolves file paths to absolute paths,
    and returns a list of valid examples with updated paths.
    """
    if dataset is None:
        logger.error("UTILS: Dataset provided to validate_and_update_dataset_paths is None.")
        return []

    processed_examples: List[Dict[str, Any]] = []
    required_keys = ['prompt', 'testbench_path', 'expected_total_tests', 'reference_verilog_path']

    for i, example_orig in enumerate(dataset):
        example = example_orig.copy()  # Work on a copy
        is_valid_example = True
        prompt_log = str(example.get('prompt', 'N/A'))[:70]

        # Check for missing required keys
        for key in required_keys:
            if key not in example or example[key] is None:
                logger.warning(f"UTILS: Dataset row {i} ('{prompt_log}...') missing or None for key '{key}'. Skipping example.")
                is_valid_example = False
                break
        if not is_valid_example:
            continue

        # Path validation and update logic
        tb_path_orig = str(example['testbench_path'])
        ref_path_orig = str(example['reference_verilog_path'])

        if dataset_base_path:
            # Resolve testbench path
            tb_full_path = os.path.join(dataset_base_path, tb_path_orig)
            if os.path.exists(tb_full_path):
                example['testbench_path'] = tb_full_path  # Update path in the copy
            else:
                logger.warning(f"UTILS: Testbench file not found for row {i} ('{prompt_log}...'): {tb_path_orig} (resolved to: {tb_full_path}). Skipping example.")
                is_valid_example = False
            
            # Resolve reference Verilog path (only if testbench was valid)
            if is_valid_example:
                ref_full_path = os.path.join(dataset_base_path, ref_path_orig)
                if os.path.exists(ref_full_path):
                    example['reference_verilog_path'] = ref_full_path  # Update path in the copy
                else:
                    logger.warning(f"UTILS: Reference Verilog file not found for row {i} ('{prompt_log}...'): {ref_path_orig} (resolved to: {ref_full_path}). Skipping example.")
                    is_valid_example = False
        else:
            # Fallback: check paths as they are (relative to CWD or absolute if already)
            if not os.path.exists(tb_path_orig):
                logger.warning(f"UTILS: Testbench file not found (dataset_base_path not provided): {tb_path_orig} for row {i} ('{prompt_log}...'). Skipping example.")
                is_valid_example = False
            
            if is_valid_example and not os.path.exists(ref_path_orig):
                logger.warning(f"UTILS: Reference Verilog file not found (dataset_base_path not provided): {ref_path_orig} for row {i} ('{prompt_log}...'). Skipping example.")
                is_valid_example = False

        if is_valid_example:
            processed_examples.append(example) # Add the modified, valid example

    if not processed_examples and len(dataset) > 0:
        logger.error("UTILS: No valid examples found after validation and path update. All examples were skipped.")
    elif len(dataset) > 0:
        logger.info(f"UTILS: Dataset validation and path update complete. {len(processed_examples)}/{len(dataset)} examples are valid and have updated paths.")
    
    return processed_examples

class DetailedWandbCallback(TrainerCallback):
    """Enhanced W&B callback with detailed reward and training metrics."""
    
    def __init__(self, env_config, script_config, reward_config, experience_buffer=None):
        self.env_config = env_config
        self.script_config = script_config
        self.reward_config = reward_config
        self.experience_buffer = experience_buffer
        self.recent_rewards = deque(maxlen=100)
        self.reward_components_history = deque(maxlen=100)
        super().__init__()

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        use_wandb = "wandb" in args.report_to if isinstance(args.report_to, list) else args.report_to == "wandb"
        if args.local_rank <= 0 and use_wandb and not self.env_config.wandb_disable:
            if wandb.run is not None:
                logger.info("Logging enhanced configs to W&B.")
                wandb.config.update({
                    "env_config": asdict(self.env_config),
                    "script_config": asdict(self.script_config),
                    "reward_config": asdict(self.reward_config)
                }, allow_val_change=True)
                
                # Log training setup
                wandb.log({
                    "setup/total_parameters": sum(p.numel() for p in kwargs.get('model', {}).parameters() if hasattr(kwargs.get('model', {}), 'parameters')),
                    "setup/trainable_parameters": sum(p.numel() for p in kwargs.get('model', {}).parameters() if p.requires_grad and hasattr(kwargs.get('model', {}), 'parameters')),
                })
            else:
                logger.warning("W&B run not initialized by Trainer at on_train_begin.")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if args.local_rank <= 0 and hasattr(wandb, 'run') and wandb.run is not None:
            try:
                # Log experience buffer statistics
                if self.experience_buffer:
                    buffer_stats = self.experience_buffer.get_stats()
                    wandb.log({
                        f"experience_buffer/{k}": v for k, v in buffer_stats.items()
                    })
                
                # Log reward distribution if available
                if self.recent_rewards:
                    wandb.log({
                        "reward_stats/mean": np.mean(self.recent_rewards),
                        "reward_stats/std": np.std(self.recent_rewards),
                        "reward_stats/max": np.max(self.recent_rewards),
                        "reward_stats/min": np.min(self.recent_rewards),
                        "reward_stats/positive_ratio": np.mean(np.array(self.recent_rewards) > 0),
                    })
                
                # Log model gradient norms if available
                if model is not None:
                    total_norm = 0
                    param_count = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    if param_count > 0:
                        total_norm = total_norm ** (1. / 2)
                        wandb.log({
                            "training/gradient_norm": total_norm,
                            "training/gradient_params": param_count
                        })
                        
            except Exception as e:
                logger.warning(f"Error in DetailedWandbCallback.on_log: {e}")

    def log_reward_components(self, reward_components: Dict[str, float]):
        """Log detailed reward component breakdown."""
        self.reward_components_history.append(reward_components)
        
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({f"reward_components/{k}": v for k, v in reward_components.items()})

    def log_reward(self, reward: float):
        """Log individual reward values."""
        self.recent_rewards.append(reward)


class StepLoggingCallback(TrainerCallback):
    """Enhanced step logging with more detailed information."""
    
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:  # Log every 10 steps to reduce noise
            logger.info(f"Starting training step {state.global_step}/{state.max_steps if state.max_steps > 0 else '∞'}")

    def on_log(self, args, state, control, **kwargs):
        if args.local_rank <= 0 and state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                logger.info(f"Step {state.global_step}: Loss = {latest_log['loss']:.6f}")


# Backward compatibility alias
InferenceCallback = EnhancedInferenceCallback