import logging
import shutil
import typing

import torch
import tqdm

from .ABCContext import Context
from .Navigator import TrainingProcessNavigator
from .TrainFileControler import TrainFileManager
from .WarpedReduceLROnPlateau import WarpedReduceLROnPlateau

TensorType = torch.Tensor


class Trainer:
    def __init__(self, ctx: Context, logger: logging.Logger, best_model_elect_score: str = "f1",
                 hyper_parameter_option_path: typing.Tuple[str, str] = ("", ""),
                 only_keep_best_epoch: bool = False):
        self.ctx = ctx
        self.logger = logger
        self.navigator = TrainingProcessNavigator(self.ctx, logger=logger)
        self.file_manager = TrainFileManager(self.ctx, logger)
        self.best_model_elect_score = best_model_elect_score
        self.only_keep_best_epoch = only_keep_best_epoch
        self.previous_saved_epoch = -1
        self.hyper_parameter_option_path = hyper_parameter_option_path
        self.logger.info(self.navigator.first_epoch_display())
        self.logger.info(f"Model selection be be performed base on {self.best_model_elect_score}")

    def one_step_loss(self, data: TensorType, label: TensorType):
        data = data.to(self.ctx.device)
        label = label.to(self.ctx.device)
        predicted = self.ctx.model(data)
        loss = self.ctx.loss_function(predicted, label)
        return loss

    def update_scheduler(self):
        if isinstance(self.ctx.scheduler, WarpedReduceLROnPlateau):
            _metrics = {
                "validate_loss": self.ctx.validate_loss[-1],
                "train_loss": self.ctx.train_loss[-1]
            }[self.ctx.scheduler.metrics_name]
            self.ctx.scheduler.step(metrics=_metrics)
        else:
            self.ctx.scheduler.step()

    def one_epoch_train(self):
        self.ctx.model.train()
        epoch_loss = torch.empty(0).to(self.ctx.device)
        for data, label in tqdm.tqdm(self.ctx.train_loader):
            loss = self.one_step_loss(data, label)
            self.ctx.optimizer.zero_grad()
            loss.backward()
            self.ctx.optimizer.step()
            epoch_loss = torch.hstack((epoch_loss, loss.detach().clone()))
        self.ctx.train_loss = torch.hstack([
            self.ctx.train_loss, mean_loss := torch.mean(epoch_loss)
        ])
        return mean_loss

    def one_epoch_validation(self):
        result, mean_loss = self.test_onset("validate")
        self.ctx.validate_loss = torch.hstack([
            self.ctx.validate_loss, mean_loss
        ])
        return result

    @torch.no_grad()
    def test_onset(self, set_type: str = "validate", reduction: str = "mean"):
        dataset = {
            "validate": self.ctx.validate_loader,
            "train": self.ctx.train_loader,
            "test": self.ctx.test_loader
        }[set_type]
        reduce_function = {
            "mean": torch.mean,
            "sum": torch.sum
        }[reduction]
        self.ctx.model.eval()
        self.ctx.tester.set_dataloader(dataset, self.ctx.n_classes)
        result = self.ctx.tester.evaluate_model()
        save_destination = {
            "validate": self.ctx.validate_score,
            "train": self.ctx.train_score,
            "test": self.ctx.test_scores
        }[set_type]

        mean_loss_of_test = reduce_function(self.ctx.tester.loss_)
        save_destination[self.ctx.current_epoch] = result
        return result, mean_loss_of_test

    def back_up_hyperparameter_and_options(self):
        try:
            backup_dir = self.ctx.dump_path
            self.logger.info(f"Backup hyperparameter and options to {backup_dir}")
            # copy "hyperparameter.py" and "options.py" to backup_dir
            hyper, opt = self.hyper_parameter_option_path
            shutil.copy2(hyper, backup_dir)
            shutil.copy2(opt, backup_dir)
        except Exception as e:
            self.logger.error(e)

    def train_process(self):
        for epoch in range(self.ctx.epochs):
            self.ctx.set_current_epoch(epoch)
            self.logger.info(f"Epoch {epoch} start.")
            self.one_epoch_train()
            validate_result = self.one_epoch_validation()
            self.update_scheduler()
            # Update the optimal validation epoch based on the chosen metric
            election_score, maximum = {
                "f1": (validate_result["f1_score"], self.ctx.maximum_f1_score),
                "acc": (validate_result["accuracy"], self.ctx.maximum_accuracy),
                "pre": (validate_result["precision"], self.ctx.maximum_precision),
                "rec": (validate_result["recall"], self.ctx.maximum_recall)
            }[self.best_model_elect_score]
            if self.file_manager.update_optimal_validation_epoch(epoch, election_score, maximum):
                self.ctx.maximum_f1_score = validate_result["f1_score"]
                self.test_onset("test")  # Test on the test set for the best model
                self.test_onset("train")

            if epoch in self.ctx.train_set_test_mile_stones:
                self.logger.info(f"Train set test milestone reached at epoch {epoch}.")
                self.test_onset("train")

            # Handle epoch file management
            self.file_manager.handle_epoch_files(epoch, self.only_keep_best_epoch)

            self.navigator.conclude_this_epoch()
            self.logger.info(f"Epoch {epoch} end.\n")

    def main(self):
        self.back_up_hyperparameter_and_options()
        try:
            self.train_process()
        except Exception as e:
            self.logger.info(e, exc_info=True)
            self.logger.info("Serializing context.")
            self.test_onset("train")
            self.test_onset("test")
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt.")
            self.logger.info("Please Wait. Test model on train, validate and test set.")
            self.test_onset("train")
            self.test_onset("validate")
            self.test_onset("test")
        finally:
            self.navigator.conclude_this_epoch()
            self.navigator.dump_metrics()
            self.logger.info(f"Train process done. last epoch data saved at {self.ctx.current_epoch} directory.")
