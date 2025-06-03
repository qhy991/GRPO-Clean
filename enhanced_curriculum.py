# enhanced_curriculum.py - 双层课程学习系统
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from datasets import Dataset
from dataclasses import dataclass
import wandb

logger = logging.getLogger(__name__)

@dataclass
class CurriculumStageConfig:
    """课程学习阶段配置"""
    name: str
    dataset_levels: List[str]           # 数据集等级过滤 ['basic', 'intermediate', 'advanced', 'expert']
    complexity_range: Tuple[float, float]  # 复杂度范围 (min, max)
    epochs_ratio: float                 # 该阶段训练epoch比例
    performance_threshold: float = 0.6  # 进入下一阶段的性能阈值
    min_evaluations: int = 5           # 最少评估次数
    description: str = ""              # 阶段描述

class EnhancedCurriculumManager:
    """增强的双层课程学习管理器"""
    
    def __init__(self, curriculum_stages: List[CurriculumStageConfig], dataset: Dataset):
        self.curriculum_stages = curriculum_stages
        self.full_dataset = dataset
        self.current_stage = 0
        self.stage_performance_history = []
        self.stage_statistics = []
        
        # 分析数据集分布
        self._analyze_dataset_distribution()
        
        # 验证课程设计
        self._validate_curriculum_design()
        
        logger.info(f"Enhanced Curriculum Manager initialized with {len(curriculum_stages)} stages")
        for i, stage in enumerate(curriculum_stages):
            logger.info(f"  Stage {i}: {stage.name} - Levels: {stage.dataset_levels}, "
                       f"Complexity: {stage.complexity_range}, Ratio: {stage.epochs_ratio}")
    def get_curriculum_state(self) -> Dict[str, Any]:
        """获取课程学习管理器的当前状态，用于保存。"""
        return {
            "current_stage": self.current_stage,  # 假设 current_stage 是阶段索引或可序列化标识
            "stage_performance_history": self.stage_performance_history,
            # 添加其他任何需要持久化的内部状态变量
            # 例如，如果你的 curriculum_stages 列表是动态生成的或修改的，也可能需要保存
        }

    def load_curriculum_state(self, state_dict: Dict[str, Any]):
        """从字典加载课程学习管理器的状态。"""
        self.current_stage = state_dict.get("current_stage", 0) # 使用 get 提供默认值以防 key 不存在
        self.stage_performance_history = state_dict.get("stage_performance_history", [])
        # 加载其他已保存的状态变量
        logger.info(f"课程学习状态已加载。从阶段 {self.current_stage} (名称: {self.curriculum_stages[self.current_stage].name if self.current_stage < len(self.curriculum_stages) else '未知'}) 恢复。")
    def get_curriculum_state(self) -> Dict[str, Any]:
        """获取课程学习管理器的当前状态，用于保存。"""
        return {
            "current_stage": self.current_stage,  # 假设 current_stage 是阶段索引或可序列化标识
            "stage_performance_history": self.stage_performance_history,
            # 添加其他任何需要持久化的内部状态变量
            # 例如，如果你的 curriculum_stages 列表是动态生成的或修改的，也可能需要保存
        }

    def load_curriculum_state(self, state_dict: Dict[str, Any]):
        """从字典加载课程学习管理器的状态。"""
        self.current_stage = state_dict.get("current_stage", 0) # 使用 get 提供默认值以防 key 不存在
        self.stage_performance_history = state_dict.get("stage_performance_history", [])
        # 加载其他已保存的状态变量
        logger.info(f"课程学习状态已加载。从阶段 {self.current_stage} (名称: {self.curriculum_stages[self.current_stage].name if self.current_stage < len(self.curriculum_stages) else '未知'}) 恢复。")

    def _analyze_dataset_distribution(self):
        """分析数据集的等级和复杂度分布"""
        if len(self.full_dataset) == 0:
            logger.warning("Empty dataset provided to curriculum manager")
            return
        
        # 统计数据集等级分布
        level_counts = {}
        complexity_by_level = {}
        
        for example in self.full_dataset:
            level = example.get('level', 'unknown').lower()
            complexity = example.get('complexity_score', 5.0)
            
            # 统计等级分布
            level_counts[level] = level_counts.get(level, 0) + 1
            
            # 统计每个等级的复杂度分布
            if level not in complexity_by_level:
                complexity_by_level[level] = []
            complexity_by_level[level].append(complexity)
        
        self.dataset_distribution = {
            'level_counts': level_counts,
            'complexity_by_level': complexity_by_level,
            'total_samples': len(self.full_dataset)
        }
        
        # 打印分布信息
        logger.info("Dataset Distribution Analysis:")
        logger.info(f"  Total samples: {self.dataset_distribution['total_samples']}")
        for level, count in level_counts.items():
            if level in complexity_by_level and complexity_by_level[level]:
                avg_complexity = np.mean(complexity_by_level[level])
                complexity_range = (np.min(complexity_by_level[level]), np.max(complexity_by_level[level]))
                logger.info(f"  {level.capitalize()}: {count} samples, "
                           f"avg complexity: {avg_complexity:.2f}, range: {complexity_range}")
    
    def _validate_curriculum_design(self):
        """验证课程设计的合理性"""
        # 检查所有数据集等级是否都被覆盖
        available_levels = set(self.dataset_distribution['level_counts'].keys())
        covered_levels = set()
        
        for stage in self.curriculum_stages:
            covered_levels.update([level.lower() for level in stage.dataset_levels])
        
        uncovered_levels = available_levels - covered_levels
        if uncovered_levels:
            logger.warning(f"Dataset levels not covered by curriculum: {uncovered_levels}")
        
        # 检查epoch比例总和
        total_ratio = sum(stage.epochs_ratio for stage in self.curriculum_stages)
        if abs(total_ratio - 1.0) > 0.01:
            logger.warning(f"Curriculum epochs ratios sum to {total_ratio:.3f}, not 1.0")
    
    def get_current_stage_dataset(self) -> Dataset:
        """获取当前阶段的数据集"""
        if self.current_stage >= len(self.curriculum_stages):
            logger.info("Curriculum completed, using full dataset")
            return self.full_dataset
        
        stage = self.curriculum_stages[self.current_stage]
        
        # 双层过滤：数据集等级 + 复杂度范围
        filtered_indices = []
        level_filter_count = 0
        complexity_filter_count = 0
        
        for i, example in enumerate(self.full_dataset):
            # 第一层过滤：数据集等级
            example_level = example.get('level', 'unknown').lower()
            if example_level not in [level.lower() for level in stage.dataset_levels]:
                continue
            level_filter_count += 1
            
            # 第二层过滤：复杂度范围
            complexity = example.get('complexity_score', 5.0)
            min_complexity, max_complexity = stage.complexity_range
            if not (min_complexity <= complexity <= max_complexity):
                continue
            complexity_filter_count += 1
            
            filtered_indices.append(i)
        
        if not filtered_indices:
            logger.warning(f"No examples found for stage {self.current_stage} ({stage.name}), using full dataset")
            return self.full_dataset
        
        stage_dataset = self.full_dataset.select(filtered_indices)
        
        # 记录统计信息
        stage_stats = {
            'stage_index': self.current_stage,
            'stage_name': stage.name,
            'total_examples': len(self.full_dataset),
            'level_filtered': level_filter_count,
            'complexity_filtered': complexity_filter_count,
            'final_selected': len(stage_dataset),
            'selection_ratio': len(stage_dataset) / len(self.full_dataset),
            'target_levels': stage.dataset_levels,
            'complexity_range': stage.complexity_range
        }
        self.stage_statistics.append(stage_stats)
        
        logger.info(f"Curriculum Stage {self.current_stage} ({stage.name}):")
        logger.info(f"  Target levels: {stage.dataset_levels}")
        logger.info(f"  Complexity range: {stage.complexity_range}")
        logger.info(f"  Selected examples: {len(stage_dataset)}/{len(self.full_dataset)} ({stage_stats['selection_ratio']:.1%})")
        logger.info(f"  Level filtering: {level_filter_count} examples passed")
        logger.info(f"  Complexity filtering: {complexity_filter_count} examples passed")
        
        return stage_dataset
    
    def should_advance_stage(self, recent_performance: float) -> bool:
        """判断是否应该进入下一阶段"""
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return False
        
        stage = self.curriculum_stages[self.current_stage]
        self.stage_performance_history.append(recent_performance)
        
        # 需要足够的评估次数
        if len(self.stage_performance_history) < stage.min_evaluations:
            logger.debug(f"Stage {self.current_stage}: Need more evaluations "
                        f"({len(self.stage_performance_history)}/{stage.min_evaluations})")
            return False
        
        # 检查最近的性能表现
        recent_window = min(3, len(self.stage_performance_history))
        recent_avg = np.mean(self.stage_performance_history[-recent_window:])
        
        should_advance = recent_avg >= stage.performance_threshold
        
        if should_advance:
            logger.info(f"Stage {self.current_stage} performance criteria met: "
                       f"recent avg = {recent_avg:.3f} >= threshold = {stage.performance_threshold}")
        else:
            logger.debug(f"Stage {self.current_stage} performance not ready: "
                        f"recent avg = {recent_avg:.3f} < threshold = {stage.performance_threshold}")
        
        return should_advance
    
    def advance_stage(self) -> bool:
        """进入下一阶段"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            # 记录当前阶段的最终统计
            final_stats = {
                'completed_stage': self.current_stage,
                'stage_name': self.curriculum_stages[self.current_stage].name,
                'total_evaluations': len(self.stage_performance_history),
                'final_performance': self.stage_performance_history[-1] if self.stage_performance_history else 0,
                'average_performance': np.mean(self.stage_performance_history) if self.stage_performance_history else 0,
                'performance_history': self.stage_performance_history.copy()
            }
            
            self.current_stage += 1
            self.stage_performance_history = []  # 重置性能历史
            
            new_stage = self.curriculum_stages[self.current_stage]
            logger.info(f"🎯 Advanced to curriculum stage {self.current_stage}: {new_stage.name}")
            logger.info(f"   Previous stage stats: {final_stats['total_evaluations']} evaluations, "
                       f"final performance: {final_stats['final_performance']:.3f}")
            logger.info(f"   New stage targets: levels {new_stage.dataset_levels}, "
                       f"complexity {new_stage.complexity_range}")
            
            return True
        
        logger.info("🏆 Curriculum learning completed! Using full dataset.")
        return False
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """获取当前阶段信息"""
        if self.current_stage >= len(self.curriculum_stages):
            return {
                'stage_index': self.current_stage,
                'stage_name': 'completed',
                'dataset_levels': 'all',
                'complexity_range': 'all',
                'is_completed': True
            }
        
        stage = self.curriculum_stages[self.current_stage]
        return {
            'stage_index': self.current_stage,
            'stage_name': stage.name,
            'dataset_levels': stage.dataset_levels,
            'complexity_range': stage.complexity_range,
            'epochs_ratio': stage.epochs_ratio,
            'performance_threshold': stage.performance_threshold,
            'current_evaluations': len(self.stage_performance_history),
            'min_evaluations': stage.min_evaluations,
            'is_completed': False
        }
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """获取整体课程进度"""
        total_stages = len(self.curriculum_stages)
        progress_ratio = (self.current_stage + 1) / total_stages if total_stages > 0 else 1.0
        
        return {
            'current_stage': self.current_stage,
            'total_stages': total_stages,
            'progress_ratio': progress_ratio,
            'completed_stages': self.current_stage,
            'stage_statistics': self.stage_statistics,
            'dataset_distribution': self.dataset_distribution
        }
    
    def log_to_wandb(self, step: int):
        """记录到W&B"""
        if not hasattr(wandb, 'run') or wandb.run is None:
            return
        
        current_info = self.get_current_stage_info()
        progress_info = self.get_curriculum_progress()
        
        # 基础信息
        wandb.log({
            'curriculum/current_stage': current_info['stage_index'],
            'curriculum/stage_name': current_info['stage_name'],
            'curriculum/progress_ratio': progress_info['progress_ratio'],
            'curriculum/completed_stages': progress_info['completed_stages'],
            'curriculum/is_completed': current_info['is_completed']
        }, step=step)
        
        # 当前阶段详细信息
        if not current_info['is_completed']:
            wandb.log({
                'curriculum/current_evaluations': current_info['current_evaluations'],
                'curriculum/min_evaluations': current_info['min_evaluations'],
                'curriculum/performance_threshold': current_info['performance_threshold'],
                'curriculum/dataset_levels': str(current_info['dataset_levels']),
                'curriculum/complexity_range': str(current_info['complexity_range'])
            }, step=step)
        
        # 性能历史
        if self.stage_performance_history:
            wandb.log({
                'curriculum/stage_performance_mean': np.mean(self.stage_performance_history),
                'curriculum/stage_performance_latest': self.stage_performance_history[-1],
                'curriculum/stage_performance_trend': np.mean(self.stage_performance_history[-3:]) if len(self.stage_performance_history) >= 3 else self.stage_performance_history[-1]
            }, step=step)


def create_default_curriculum_stages() -> List[CurriculumStageConfig]:
    """创建默认的双层课程学习阶段"""
    stages = [
        CurriculumStageConfig(
            name="foundation",
            dataset_levels=["basic"],
            complexity_range=(0.0, 3.0),
            epochs_ratio=0.25,
            performance_threshold=0.7,
            min_evaluations=5,
            description="基础阶段：学习简单的基础级设计"
        ),
        CurriculumStageConfig(
            name="elementary",
            dataset_levels=["basic", "intermediate"],
            complexity_range=(0.0, 5.0),
            epochs_ratio=0.25,
            performance_threshold=0.65,
            min_evaluations=5,
            description="初级阶段：基础级+简单中级设计"
        ),
        CurriculumStageConfig(
            name="intermediate",
            dataset_levels=["intermediate"],
            complexity_range=(3.0, 7.0),
            epochs_ratio=0.25,
            performance_threshold=0.6,
            min_evaluations=4,
            description="中级阶段：中等复杂度的中级设计"
        ),
        CurriculumStageConfig(
            name="advanced",
            dataset_levels=["intermediate", "advanced"],
            complexity_range=(5.0, 9.0),
            epochs_ratio=0.15,
            performance_threshold=0.55,
            min_evaluations=4,
            description="高级阶段：复杂的中级和高级设计"
        ),
        CurriculumStageConfig(
            name="expert",
            dataset_levels=["advanced", "expert"],
            complexity_range=(7.0, 10.0),
            epochs_ratio=0.1,
            performance_threshold=0.5,
            min_evaluations=3,
            description="专家阶段：最复杂的高级和专家级设计"
        )
    ]
    return stages


def create_custom_curriculum_stages(
    dataset_distribution: Dict[str, Any],
    focus_levels: List[str] = None,
    complexity_emphasis: str = "balanced"  # "simple", "balanced", "complex"
) -> List[CurriculumStageConfig]:
    """根据数据集分布创建自定义课程阶段"""
    
    if focus_levels is None:
        focus_levels = ["basic", "intermediate", "advanced", "expert"]
    
    # 根据复杂度偏好调整复杂度范围
    if complexity_emphasis == "simple":
        complexity_ranges = [(0, 3), (0, 4), (2, 6), (4, 8), (6, 10)]
    elif complexity_emphasis == "complex":
        complexity_ranges = [(0, 4), (2, 6), (4, 8), (6, 10), (8, 10)]
    else:  # balanced
        complexity_ranges = [(0, 3), (0, 5), (3, 7), (5, 9), (7, 10)]
    
    stages = []
    
    # 基础阶段
    if "basic" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="foundation",
            dataset_levels=["basic"],
            complexity_range=complexity_ranges[0],
            epochs_ratio=0.3,
            performance_threshold=0.7,
            description="基础阶段：最简单的基础级设计"
        ))
    
    # 初级阶段
    if "basic" in focus_levels and "intermediate" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="elementary",
            dataset_levels=["basic", "intermediate"],
            complexity_range=complexity_ranges[1],
            epochs_ratio=0.25,
            performance_threshold=0.65,
            description="初级阶段：基础到中级的过渡"
        ))
    
    # 中级阶段
    if "intermediate" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="intermediate",
            dataset_levels=["intermediate"],
            complexity_range=complexity_ranges[2],
            epochs_ratio=0.25,
            performance_threshold=0.6,
            description="中级阶段：中等复杂度设计"
        ))
    
    # 高级阶段
    if "advanced" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="advanced",
            dataset_levels=["intermediate", "advanced"],
            complexity_range=complexity_ranges[3],
            epochs_ratio=0.15,
            performance_threshold=0.55,
            description="高级阶段：复杂设计"
        ))
    
    # 专家阶段
    if "expert" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="expert",
            dataset_levels=["advanced", "expert"],
            complexity_range=complexity_ranges[4],
            epochs_ratio=0.05,
            performance_threshold=0.5,
            description="专家阶段：最复杂设计"
        ))
    
    # 标准化epoch比例
    total_ratio = sum(stage.epochs_ratio for stage in stages)
    for stage in stages:
        stage.epochs_ratio /= total_ratio
    
    return stages