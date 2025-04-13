import os
import typing

import torch
import torch.nn as nn
import torch.optim as optimize
import torch.utils.data as tch_data

from MuxkitTools.score_tools.ClassifierTester import MonoLabelClassificationTester
from esc50.ESC50IO import ESC50, label_balanced_split_esc50, get_index_to_category
from src.ABCContext import Context
from src.PreprocessedDataset import create_cached_preprocessed_dataset
from . import hyperparameters as hyp
from . import options as opt
from .DataPreprocessor import DataPreprocessor
from .model import make_model


class TrainContext(Context):  # 继承自 ABCContext
    def __init__(self, serial: typing.Optional[typing.Dict[str, typing.Any]] = None):
        super().__init__()
        # 设置基本参数
        (self.set_device(opt.Device)
         .set_epochs(hyp.Epochs)
         .set_dump_path(opt.SaveDirectory)
         .set_n_classes(hyp.N_Classes)
         .set_compile_model(opt.CompileModel)
         .set_context_identifier(f"resnet_{opt.TrainID}{'_dryrun' if opt.DryRun else ''}"))

        # 完成全量初始化
        self.full_initialization()

        # 如果存在序列化数据则加载
        if serial is not None:
            self.unserialize(serial)

    def initialize_model(self):
        # 使用 ResNet 初始化模型
        self.model: torch.nn.Module = make_model(hyp.ResNet, n_cls=hyp.N_Classes).to(self.device)
        if self.compile_model and self.model is not None:
            self.model = torch.compile(self.model)

    def initialize_loss(self):
        # 使用交叉熵损失函数
        self.loss_function: torch.nn.Module = nn.CrossEntropyLoss()

    def initialize_optimizer(self):
        # 初始化 Adam 优化器
        self.optimizer: optimize.Optimizer = optimize.Adam(
            self.model.parameters(), lr=hyp.LearningRate
        )

    def initialize_scheduler(self):
        # 使用 CosineAnnealingWarmRestarts 作为调度器
        self.scheduler: optimize.lr_scheduler.LRScheduler = optimize.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, hyp.T_0, hyp.T_mult, hyp.eta_min
        )

    def initialize_datasets(self):
        # 初始化数据集和数据加载器
        self.data_preprocessor = DataPreprocessor().to(self.device)
        self.dataset_path = opt.DataSetPath

        dataset = ESC50(self.dataset_path, download=(not os.path.exists(self.dataset_path)), transform=None,
                        subset="all")
        train_set, validate_set, test_set = label_balanced_split_esc50(dataset, [0.7, 0.15, 0.15])

        self.train_set, self.validate_set, self.test_set = [
            create_cached_preprocessed_dataset(self.data_preprocessor, i_set, opt.CacheSize, self.device, hyp.N_Classes)
            for i_set in [train_set, validate_set, test_set]
        ]

        self.train_loader, self.validate_loader, self.test_loader = [
            tch_data.DataLoader(i_set, batch_size=hyp.BatchSize)
            for i_set in [self.train_set, self.validate_set, self.test_set]
        ]

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
