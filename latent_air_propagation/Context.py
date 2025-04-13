import os
import typing

import torch
import torch.nn as nn
import torch.optim as optimize
import torch.utils.data as tch_data

from lib.MuxkitTools.score_tools.ClassifierTester import MonoLabelClassificationTester
from lib.esc50_io.ESC50IO import get_index_to_category
from src.ABCContext import Context
from . import hyperparameters as hyp
from . import options as opt
from .model import make_model
from src.LatentDataset import DataTensorDataset
from src.PreprocessedDataset import create_cached_preprocessed_dataset, create_preprocessed_acoustic_dataset
from . import DataPreprocessor
from lib.MuxkitTools.audio_tools.bc_augmentation.bc_augmented_dataset import BCLearningDataset
from lib.MuxkitTools.model_tools.stati import stati_model
from src.WarpedReduceLROnPlateau import WarpedReduceLROnPlateau
from src.SoftmaxLogKLDivLoss import SoftmaxLogKLDivLoss
class TrainContext(Context):  # 继承自 ABCContext


    def __init__(self, serial: typing.Optional[typing.Dict[str, typing.Any]] = None):
        super().__init__()
        (self.set_device(opt.Device)
         .set_epochs(hyp.Epochs)
         .set_dump_path(opt.SaveDirectory)
         .set_n_classes(hyp.N_Classes)
         .set_compile_model(opt.CompileModel)
         .set_context_identifier(f"resnet_{opt.TrainID}{'_dryrun' if opt.DryRun else ''}"))
        self.full_initialization()
        if serial is not None:
            self.unserialize(serial)

    def initialize_model(self):
        # 使用 ResNet 初始化模型
        self.model: torch.nn.Module = make_model(hyp.ResNet, n_cls=hyp.N_Classes).to(self.device)
        if self.compile_model and self.model is not None:
            self.model = torch.compile(self.model)

    def initialize_loss(self):
        # 使用交叉熵损失函数
        self.loss_function: torch.nn.Module = SoftmaxLogKLDivLoss()


    def initialize_optimizer(self):
        # 初始化 Adam 优化器
        self.optimizer: optimize.Optimizer = optimize.Adam(
            self.model.parameters(), lr=hyp.LearningRate, weight_decay=hyp.WeightDecay
        )

    def initialize_scheduler(self):
        self.scheduler: optimize.lr_scheduler.LRScheduler = WarpedReduceLROnPlateau(
            self.optimizer, mode=hyp.ReduceLROnPlateauMode,
            factor=hyp.ReduceLROnPlateauFactor,
            patience=hyp.ReduceLROnPlateauPatience,
            metrics_name=hyp.ReduceLROnPlateauMetricsName,
            min_lr=hyp.ReduceLROnPlateauMinLR
        )
    def initialize_datasets(self):
        train_set = DataTensorDataset.from_datatensor_path(
            split="train",
            data_tensor_path=opt.TrainSetPath.resolve(),
            device=self.device
        )
        test_set = DataTensorDataset.from_datatensor_path(
            split="test",
            data_tensor_path=opt.TestSetPath.resolve(),
            device=self.device
        )
        validate_set = DataTensorDataset.from_datatensor_path(
            split="validate",
            data_tensor_path=opt.ValidatePath.resolve(),
            device=self.device
        )
        self.data_preprocessor = DataPreprocessor.DataPreprocessor().to(self.device)

        train_set = BCLearningDataset(
            train_set, hyp.AudioSampleRate, hyp.N_Classes, self.device
        )

        self.train_set = create_preprocessed_acoustic_dataset(
            self.data_preprocessor,
            train_set,
            self.device,
            hyp.N_Classes,
            True
        )

        self.test_set = create_cached_preprocessed_dataset(
            self.data_preprocessor,
            test_set,
            opt.CacheSize,
            self.device,
            hyp.N_Classes,
            True
        )
        self.validate_set = create_cached_preprocessed_dataset(
            self.data_preprocessor,
            validate_set,
            opt.CacheSize,
            self.device,
            hyp.N_Classes,
            True
        )

        self.train_loader = tch_data.DataLoader(self.train_set, shuffle=True, batch_size=hyp.BatchSize)
        self.validate_loader = tch_data.DataLoader(self.validate_set, shuffle=False, batch_size=hyp.BatchSize)
        self.test_loader = tch_data.DataLoader(self.test_set, shuffle=False, batch_size=hyp.BatchSize)

    def initialize_tester(self):
        # 初始化测试器
        self.tester = MonoLabelClassificationTester(self.model, self.device, self.loss_function)

    def init_other(self):
        self.classindex_to_label = get_index_to_category()
        self.train_set_test_mile_stones = hyp.TrainSetTestMileStone

    def initialize_loss_trackers(self):
        # 初始化损失跟踪器
        self.validate_loss = torch.empty(0).to(self.device)
        self.train_loss = torch.empty(0).to(self.device)

    def summary(self):
        """
        生成训练摘要信息。
        """
        if self.summary_content is None:
            summary = ["TrainContext Summary", f"Model:\n{str(self.model)}", f"Random seed: {self.random_seed}",
                       f"Dataset sizes - Train: {len(self.train_set)}, Validate: {len(self.validate_set)}, Test: {len(self.test_set)}"]

            # 数据集信息

            # 数据输入输出形状
            d0, l0 = self.train_set[0]
            bd0, bl0 = next(iter(self.train_loader))
            summary.append(f"Input data shape: {d0.shape}, batch shape: {bd0.shape}")
            summary.append(f"Label shape: {l0.shape}, batch shape: {bl0.shape}")

            # 模型统计信息
            model_stats = stati_model(self.model, unit="mb")
            summary += [
                f'model_param_count_with_grad: {model_stats["param_count_with_grad"]}',
                f'model_param_count_without_grad: {model_stats["param_count_without_grad"]}',
                f'model_size_mb: {model_stats["model_size"]}',
            ]

            # 训练参数
            summary.append(f"Training parameters - Batch size: {hyp.BatchSize}, Learning rate: {hyp.LearningRate}, "
                           f"Weight decay: {hyp.WeightDecay}, Scheduler: {self.scheduler.__class__.__name__}")
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