import os
import typing

import torch
import torch.nn as nn
import torch.optim as optimize
import torch.utils.data as tch_data

from lib.MuxkitTools.score_tools.ClassifierTester import MonoLabelClassificationTester
from src.ABCContext import Context
from src.BCAugmentationDataset import BCLearningDataset
from src.PreprocessedDataset import create_cached_preprocessed_dataset, CacheableDataset, \
    create_preprocessed_acoustic_dataset
from src.SoftmaxLogKLDivLoss import SoftmaxLogKLDivLoss
from . import hyperparameters as hyp
from . import options as opt
from lib.MuxkitTools.model_tools.stati import stati_model
from src.WarpedReduceLROnPlateau import WarpedReduceLROnPlateau


class TrainContext(Context):  # 继承自 ABCContext
    def __init__(self, serial: typing.Optional[typing.Dict[str, typing.Any]] = None):
        super().__init__()
        # 设置训练的主要参数
        self.set_device(opt.Device) \
            .set_epochs(hyp.Epochs) \
            .set_n_classes(hyp.N_Classes) \
            .set_compile_model(opt.CompileModel) \
            .set_context_identifier(f"{hyp.Identifier}") \
            .set_random_seed(hyp.RandomSeed)  # 设置随机种子
        self.set_dump_path(opt.SaveDirectory + self.context_identifier + f"-d-model-{hyp.d_model}")

        # 初始化其他必要模块
        self.full_initialization()  # 使用 full_initialization 方法调用各初始化方法

        # 载入序列化数据（如果有）
        if serial is not None:
            self.unserialize(serial)


    def summary(self):
        """
        生成训练摘要信息。
        """
        if self.summary_content is None:
            summary = ["TrainContext Summary"]
            summary.append(f"Model:\n{str(self.model)}")
            summary.append(f"Random seed: {self.random_seed}")

            # 数据集信息
            summary.append(
                f"Dataset sizes - Train: {len(self.train_set)}, Validate: {len(self.validate_set)}, Test: {len(self.test_set)}")

            # 数据输入输出形状
            d0, l0 = self.train_set[0]
            bd0, bl0 = next(iter(self.train_loader))
            summary.append(f"Input data shape: {d0.shape}, batch shape: {bd0.shape}")
            summary.append(f"Label shape: {l0.shape}, batch shape: {bl0.shape}")

            # 模型统计信息
            model_stats = stati_model(self.model, unit="mb")
            embedding_stats = stati_model(self.model.get_submodule("embedding"), unit="mb")
            transformer_stats = stati_model(self.model.get_submodule("transformer"), unit="mb")
            summary += [
                f'model_param_count_with_grad: {model_stats["param_count_with_grad"]}',
                f'model_param_count_without_grad: {model_stats["param_count_without_grad"]}',
                f'model_size_mb: {model_stats["model_size"]}',
                f'embedding_param_count_with_grad: {embedding_stats["param_count_with_grad"]}',
                f'embedding_param_count_without_grad: {embedding_stats["param_count_without_grad"]}',
                f'embedding_size_mb: {embedding_stats["model_size"]}',
                f'transformer_param_count_with_grad: {transformer_stats["param_count_with_grad"]}',
                f'transformer_param_count_without_grad: {transformer_stats["param_count_without_grad"]}',
                f'transformer_size_mb: {transformer_stats["model_size"]}'
            ]

            # 训练参数
            summary.append(f"Training parameters - Batch size: {hyp.BatchSize}, Learning rate: {hyp.LearningRate}, "
                           f"Scheduler: {self.scheduler.__class__.__name__}")
            summary.append(
                f"Loss function: {self.loss_function.__class__.__name__}"
            )
            # 训练进度
            summary.append(f"Training progress - Current epoch: {self.current_epoch}/{self.epochs}")

            # 记录的性能指标
            summary.append(
                f"Best metrics so far - F1 Score: {self.maximum_f1_score:.4f}, Accuracy: {self.maximum_accuracy:.4f}, "
                f"Recall: {self.maximum_recall:.4f}, Precision: {self.maximum_precision:.4f} (at epoch {self.best_model_occ_epoch})")

            self.summary = summary
        return "\n".join(self.summary)
