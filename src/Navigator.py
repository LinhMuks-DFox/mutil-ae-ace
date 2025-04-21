import csv
import json
import logging
import os
import typing

import matplotlib.pyplot as plt

from lib.MuxkitTools.plot_tools.SklearnConfusionMatrixPlotter import ConfusionMatrixPlotter
from lib.MuxkitTools.score_tools.ClassifierTester import MetricsStatusMap
from src.ABCContext import Context

_METRICS_REPORT_MSG_FMT = (
    "{dataset} F1:                {f1:<10.5g}\n"
    "{dataset} Accuracy:          {acc:<10.5g}\n"
    "{dataset} Precision:         {pre:<10.5g}\n"
    "{dataset} Recall:            {rec:<10.5g}\n"
)


class TrainingProcessNavigator:

    def __init__(self, context: Context, logger: logging.Logger):
        self.ctx = context
        self.plotter = ConfusionMatrixPlotter(
            self.ctx.classindex_to_label
        )
        self.save_dpi = 300
        self.logger = logger

    def first_epoch_display(self):
        return self.ctx.summary()

    def plot_losses(self):
        train_loss = self.ctx.train_loss.detach().clone()
        validate_loss = self.ctx.validate_loss.detach().clone()
        max_clip = self.ctx.visualaization_loss_clamp or 10.0
        train_loss = train_loss.clamp(max=max_clip).cpu().numpy()
        validate_loss = validate_loss.clamp(max=max_clip).cpu().numpy()
        fig, ax = plt.subplots()
        ax.set_title(f"{self.ctx.current_epoch} Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(train_loss, label="Train Loss")
        ax.plot(validate_loss, label="Validation Loss")
        ax.legend()
        return fig

    def formatted_epoch_message(self):
        # 生成基础信息
        message = (
            f"Epoch:                  {self.ctx.current_epoch} / {self.ctx.epochs}\n"
            f"Train Loss:             {f'{self.ctx.train_loss[-1].item():.4f}' if len(self.ctx.train_loss) > 0 else 'N/A'}\n"
            f"Validate Loss:          {f'{self.ctx.validate_loss[-1].item():.4f}' if len(self.ctx.validate_loss) > 0 else 'N/A'}\n"
            f"Learning Rate:          {self.ctx.optimizer.param_groups[0]['lr']}\n"
        )

        # 验证集信息
        if self.ctx.current_epoch in self.ctx.validate_score:
            validate_score = self.ctx.validate_score[self.ctx.current_epoch]
            message += _METRICS_REPORT_MSG_FMT.format(
                dataset="Validate",
                f1=validate_score["f1_score"],
                acc=validate_score["accuracy"],
                pre=validate_score["precision"],
                rec=validate_score["recall"],
            )

            # 比较与前一轮的差异
            if self.ctx.current_epoch != 0:
                previous_validate_score = self.ctx.validate_score[self.ctx.current_epoch - 1]
                d_f1, d_acc, d_pre, d_rec = (
                    validate_score["f1_score"] - previous_validate_score["f1_score"],
                    validate_score["accuracy"] - previous_validate_score["accuracy"],
                    validate_score["precision"] - previous_validate_score["precision"],
                    validate_score["recall"] - previous_validate_score["recall"],
                )

                message += (
                    "Compare to previous validate:\n"
                    f"F1, ACC, PRE, REC: {d_f1:,.3g}, {d_acc:,.3g}, {d_pre:,.3g}, {d_rec:,.3g}\n"
                )

        # 测试集信息
        if self.ctx.current_epoch in self.ctx.test_scores:
            test_score = self.ctx.test_scores[self.ctx.current_epoch]
            message += _METRICS_REPORT_MSG_FMT.format(
                dataset="Test",
                f1=test_score["f1_score"],
                acc=test_score["accuracy"],
                pre=test_score["precision"],
                rec=test_score["recall"],
            )

        # 训练集信息
        if self.ctx.current_epoch in self.ctx.train_score:
            train_score = self.ctx.train_score[self.ctx.current_epoch]
            message += _METRICS_REPORT_MSG_FMT.format(
                dataset="Train",
                f1=train_score["f1_score"],
                acc=train_score["accuracy"],
                pre=train_score["precision"],
                rec=train_score["recall"],
            )

        return message

    def check_serialize_context(self, current_epoch):
        return (current_epoch in self.ctx.train_score.keys() or
                current_epoch in self.ctx.test_scores.keys() or
                current_epoch == self.ctx.epochs - 1)  # The last epoch

    def _save_fig_in_formats(
            self,
            fig: plt.Figure,
            dump_path: str,
            filename_prefix: str,
            dpi: int = None,
            save_pdf: bool = True,
            save_png: bool = True
    ):
        """
        A helper function to save the figure in different formats.
        Args:
            fig (plt.Figure): The figure to be saved.
            dump_path (str): The directory path where the figure will be saved.
            filename_prefix (str): The prefix (or base name) of the output file.
            dpi (int, optional): The DPI for saving. If None, uses self.save_dpi.
            save_pdf (bool): Whether to save in PDF format.
            save_png (bool): Whether to save in PNG format.
        """
        dpi = dpi or self.save_dpi

        if save_pdf:
            fig.savefig(os.path.join(dump_path, f"{filename_prefix}.pdf"), dpi=dpi)

        if save_png:
            fig.savefig(os.path.join(dump_path, f"{filename_prefix}.png"), dpi=dpi)

        plt.close(fig)

    def conclude_this_epoch(self) -> str:
        epoch_message = self.formatted_epoch_message()
        current_epoch = self.ctx.current_epoch
        self.logger.info(epoch_message)
        serialize_context = self.check_serialize_context(current_epoch)
        if not serialize_context:
            return epoch_message

        _, dump_path = self.ctx.serialize()

        loss_fig = self.plot_losses()
        self._save_fig_in_formats(loss_fig, dump_path, "loss")  # 默认保存pdf和png

        validate_cfx = self.plotter.plot(
            self.ctx.validate_score[current_epoch]["confusion_matrix"]
        )
        self._save_fig_in_formats(validate_cfx, dump_path, "validate_confusion_matrix_fig")

        if current_epoch in self.ctx.train_score.keys():
            train_cfx = self.plotter.plot(
                self.ctx.train_score[current_epoch]["confusion_matrix"]
            )
            self._save_fig_in_formats(train_cfx, dump_path, "train_confusion_matrix_fig")
        if current_epoch in self.ctx.test_scores.keys():
            test_cfx = self.plotter.plot(
                self.ctx.test_scores[current_epoch]["confusion_matrix"]
            )
            self._save_fig_in_formats(test_cfx, dump_path, "test_confusion_matrix_fig")
        return epoch_message

    def _dump_score_to_json(self):
        # 将train_score, test_scores, validate_score中的confusion_matrix转成列表格式
        for score_dict, path in zip(
                [self.ctx.train_score, self.ctx.test_scores, self.ctx.validate_score],
                ["train", "test", "validate"]
        ):
            # 创建新的字典来存放可序列化的数据
            serializable_dict = {}
            for epoch, metrics in score_dict.items():
                # 深拷贝当前的metrics字典并转换confusion_matrix
                serializable_metrics = metrics.copy()
                serializable_metrics["confusion_matrix"] = metrics[
                    "confusion_matrix"].tolist() if "confusion_matrix" in metrics else None
                serializable_dict[epoch] = serializable_metrics

            # 保存为JSON文件
            with open(os.path.join(self.ctx.dump_path, f"{path}.json"), "w") as f:
                json.dump(serializable_dict, f, indent=4)

    def _dump_scores_to_csv(self):
        """保存为CSV文件，便于人类阅读."""
        for score_dict, path in zip(
                [self.ctx.train_score, self.ctx.test_scores, self.ctx.validate_score],
                ["train", "test", "validate"]
        ):
            csv_path = os.path.join(self.ctx.dump_path, f"{path}.csv")
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(["Epoch", "F1 Score", "Accuracy", "Precision", "Recall"])

                # 写入每个 epoch 的数据
                for epoch, metrics in sorted(score_dict.items()):
                    writer.writerow([
                        epoch,
                        metrics["f1_score"],
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                    ])

    def _dump_metrics_plot(self):
        for score, name in zip(
                [self.ctx.train_score, self.ctx.test_scores, self.ctx.validate_score],
                ["train", "test", "validate"]
        ):
            fig, ax = self._dump_score_plot(score)
            self._save_fig_in_formats(fig, self.ctx.dump_path, f"{name}_metrics_trends")

    def _dump_score_plot(self, score: dict):
        fig, ax = plt.subplots()
        epochs = sorted(score.keys())
        f1_scores = [score[epoch]["f1_score"] for epoch in epochs]
        accuracies = [score[epoch]["accuracy"] for epoch in epochs]
        precisions = [score[epoch]["precision"] for epoch in epochs]
        recalls = [score[epoch]["recall"] for epoch in epochs]

        # 绘制每个指标
        ax.plot(epochs, f1_scores, label="F1 Score")
        ax.plot(epochs, accuracies, label="Accuracy")
        ax.plot(epochs, precisions, label="Precision")
        ax.plot(epochs, recalls, label="Recall")

        # 设置图形标题、标签等
        ax.set_title("Metrics Trends Over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend()
        return fig, ax

    def dump_metrics(self):
        self._dump_score_to_json()
        self._dump_metrics_plot()
        self._dump_scores_to_csv()

    def query_best_epoch(self, sort_by: str = "f1") -> typing.Tuple[int, MetricsStatusMap]:
        """
        Query the epoch with the best performance based on the given metric.

        Args:
            sort_by (str): The metric to sort by. Can be "f1", "acc", "pre", or "rec".

        Returns:
            Tuple[int, MetricsStatusMap]: The best epoch and its corresponding metrics.
        """
        metric_mapping = {
            "f1": "f1_score",
            "acc": "accuracy",
            "pre": "precision",
            "rec": "recall",
        }

        if sort_by not in metric_mapping:
            raise ValueError(f"sort_by must be one of {list(metric_mapping.keys())}, got {sort_by}")

        metric_name = metric_mapping[sort_by]

        if not self.ctx.validate_score:
            self.logger.warning("No validation scores available.")
            return -1, None

        # Extract all validation scores
        scores = [
            (epoch, metrics_map)
            for epoch, metrics_map in self.ctx.validate_score.items()
        ]

        # Sort epochs based on the specified metric
        best_epoch, best_metrics = max(
            scores, key=lambda x: getattr(x[1], metric_name)
        )

        self.logger.info(f"Best epoch by {sort_by}: {best_epoch} with score {getattr(best_metrics, metric_name)}")
        return best_epoch, best_metrics
