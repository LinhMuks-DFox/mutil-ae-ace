import logging
import os
import shutil

from src.ABCContext import Context


class TrainFileManager:

    def __init__(self, context: Context, logger: logging.Logger):
        self.ctx = context
        self.logger = logger
        self.last_optimal_validation_epoch: int = -1
        self.last_saved_epoch: int = -1

    def delete_epoch_save_file_by_index(self, epoch_index: int):
        if epoch_index < 0:
            return
        path = self.ctx.compose_dump_path(epoch_index)
        if os.path.exists(path):
            self.logger.info(f"Removing directory for epoch {epoch_index}: {path}")
            shutil.rmtree(path)
        else:
            self.logger.warning(f"Path not found for epoch {epoch_index}: {path}")

    def update_optimal_validation_epoch(self, epoch_index: int, current_score: float, best_score: float):
        if current_score > best_score:
            # Delete previous best epoch files if they exist
            if self.last_optimal_validation_epoch >= 0:
                self.delete_epoch_save_file_by_index(self.last_optimal_validation_epoch)
            self.last_optimal_validation_epoch = epoch_index
            self.logger.info(f"New optimal validation epoch: {epoch_index} with score {current_score}")
            return True
        return False

    def handle_epoch_files(self, current_epoch: int, only_keep_best_epoch: bool):
        """
        Manage which files to keep and delete based on the current epoch and saving policy.
        """
        if only_keep_best_epoch:
            # Delete the previously saved non-optimal epoch (if any)
            if self.last_saved_epoch >= 0 and self.last_saved_epoch != self.last_optimal_validation_epoch:
                self.delete_epoch_save_file_by_index(self.last_saved_epoch)
            self.last_saved_epoch = current_epoch

    def get_summary(self):
        return {
            "last_optimal_validation_epoch": self.last_optimal_validation_epoch,
            "last_saved_epoch": self.last_saved_epoch,
        }
