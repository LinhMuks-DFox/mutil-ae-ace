from MuxkitTools.score_tools.ClassifierTester import MetricsStatusMap, ClassifierTester
import json
import os
import typing
from abc import ABC, abstractmethod
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tch_data
from MuxkitTools.train_tools.utils import set_manual_seed

OptionalModule = typing.Optional[torch.nn.Module]
DataPreprocessorType = typing.Callable[[torch.Tensor, ], torch.Tensor]


class Context(ABC):
    summary_content: str | None
    random_seed: int | None
    context_identifier: str | None
    device: torch.device | str
    compile_model: bool
    model: Optional[nn.Module] | typing.Callable
    loss_function: Optional[nn.Module]
    optimizer: Optional[optim.Optimizer]
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
    train_loader: Optional[tch_data.DataLoader]
    validate_loader: Optional[tch_data.DataLoader]
    test_loader: Optional[tch_data.DataLoader]
    train_loss: torch.Tensor
    validate_loss: torch.Tensor
    dump_path: str
    epochs: int
    current_epoch: int
    maximum_f1_score: float
    maximum_accuracy: float
    maximum_recall: float
    maximum_precision: float
    best_model_occ_epoch: int
    train_score: Dict[int, MetricsStatusMap]
    validate_score: Dict[int, MetricsStatusMap]
    test_scores: Dict[int, MetricsStatusMap]
    n_classes: int
    train_set_test_mile_stones: typing.List[int] | None
    classindex_to_label: typing.Dict[int, str]
    tester: ClassifierTester | None
    data_preprocessor: DataPreprocessorType | None
    dataset_path: str | None
    train_set: torch.utils.data.Dataset | None
    validate_set: torch.utils.data.Dataset | None
    test_set: torch.utils.data.Dataset | None

    def __init__(self):
        self.device = "cpu"
        self.best_model_occ_epoch = -1
        self.compile_model = False
        self.model = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.validate_loader = None
        self.test_loader = None
        self.train_loss = torch.empty(0)
        self.validate_loss = torch.empty(0)
        self.dump_path = ""
        self.epochs = 0
        self.current_epoch = 0
        self.maximum_f1_score = 0.
        self.maximum_accuracy = 0.
        self.maximum_recall = 0.
        self.maximum_precision = 0.
        self.train_score = {}
        self.validate_score = {}
        self.test_scores = {}
        self.n_classes = 0
        self.train_set_test_mile_stones = None
        self.tester = None
        self.summary_content = None
        self.random_seed = None

    def set_device(self, device: torch.device | str):
        self.device = device
        return self

    def set_context_identifier(self, identifier: str):
        self.context_identifier = identifier
        return self

    def set_compile_model(self, compile_model: bool):
        self.compile_model = compile_model
        return self

    def set_dump_path(self, dump_path: str):
        self.dump_path = dump_path
        return self

    def set_epochs(self, epochs: int):
        self.epochs = epochs
        return self

    def set_n_classes(self, n_classes: int):
        self.n_classes = n_classes
        return self

    def set_current_epoch(self, current_epoch: int):
        self.current_epoch = current_epoch
        return self

    def set_random_seed(self, seed: int):
        self.random_seed = seed
        set_manual_seed(self.random_seed)
        return self

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def initialize_loss(self):
        pass

    @abstractmethod
    def initialize_optimizer(self):
        pass

    @abstractmethod
    def initialize_scheduler(self):
        pass

    @abstractmethod
    def initialize_datasets(self):
        pass

    @abstractmethod
    def initialize_tester(self):
        pass

    @abstractmethod
    def initialize_loss_trackers(self):
        pass

    def full_initialization(self):
        self.initialize_model()
        self.initialize_loss_trackers()
        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_datasets()
        self.initialize_tester()
        self.init_other()

    @abstractmethod
    def init_other(self):
        pass

    def serialize(self):
        dump_path = self.compose_dump_path(self.current_epoch)
        os.makedirs(dump_path, exist_ok=True)

        model_save_path = os.path.join(dump_path, f"model_epoch_{self.current_epoch}.pt")
        optimizer_save_path = os.path.join(dump_path, f"optimizer_epoch_{self.current_epoch}.pt")
        scheduler_save_path = os.path.join(dump_path, f"scheduler_epoch_{self.current_epoch}.pt")

        torch.save(self.model.state_dict(), model_save_path)
        torch.save(self.optimizer.state_dict(), optimizer_save_path)
        torch.save(self.scheduler.state_dict(), scheduler_save_path)

        serial_data = {
            "model_path": model_save_path,
            "optimizer_path": optimizer_save_path,
            "scheduler_path": scheduler_save_path,
            "current_epoch": self.current_epoch,
            "max_f1_score": self.maximum_f1_score
        }

        json_save_path = os.path.join(dump_path, f"train_context_epoch_{self.current_epoch}.json")
        with open(json_save_path, "w") as f:
            json.dump(serial_data, f, indent=4)

        return serial_data, dump_path

    def compose_dump_path(self, epoch: int):
        return os.path.join(self.dump_path, f"epoch{epoch}")

    def unserialize(self, json_obj):
        model_path = json_obj["model_path"]
        optimizer_path = json_obj["optimizer_path"]
        scheduler_path = json_obj["scheduler_path"]

        self.model.load_state_dict(torch.load(model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        self.scheduler.load_state_dict(torch.load(scheduler_path))

        self.current_epoch = json_obj["current_epoch"]

    @abstractmethod
    def summary(self):
        pass
