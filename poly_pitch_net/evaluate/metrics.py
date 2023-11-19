import torch
import torchutil

import poly_pitch_net as ppn

# Evaluation threshold for RPA and RCA
THRESHOLD = 50  # cents

class Metrics:
    def __init__(self, threshold):
        self.accuracy = Accuracy(threshold)
        self.rmse = RMSE()
        self.rpa = RPA(threshold)

        self.reset()

    def update(self, predicted: torch.tensor, target: torch.tensor):
        self.accuracy.update(predicted, target)
        self.rmse.update(predicted, target)
        self.rpa.update(predicted, target)

    def get_metrics(self):
        metrics = {}
        metrics["accuracy"] = self.accuracy()
        metrics["RMSE"] = self.rmse()
        metrics["RPA"] = self.rpa()

        return metrics

    def reset(self):
        self.accuracy.reset()
        self.rmse.reset()
        self.rpa.reset()


class Accuracy(torchutil.metrics.Accuracy):
    """Batch-updating accuracy metric with a threshold"""

    def __init__(self, threshold: int) -> None:
        super().__init__()
        self.threshold = threshold

    def __call__(self) -> float:
        """Retrieve the current accuracy value

            The current accuracy value
        """
        return (self.true_positives / self.count)
    
    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update accuracy
        Args:
            predicted (torch.Tensor)
                Predicted pitch in cents
            target (torch.Tensor)
                Target pitch in cents

        Both predicted and target should include 0, where there's no pitch.
        """
        predicted_with_threshold = (target - predicted).abs() <= self.threshold
        positives_with_threshold = predicted[predicted_with_threshold]

        self.count += predicted.numel()
        self.true_positives += positives_with_threshold.numel()


class Precision(torchutil.metrics.Precision):
    """Bath-updating precision meric with a threshold"""

    def __init__(self, threshold: int) -> None:
        super().__init__()
        self.threshold = threshold

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric

        Arguments
            predicted
                The model prediction of pitch in cents
            target
                The corresponding ground truth of pitch in cents

        Both predicted and target should include 0, where there's no pitch.
        """
        predicted_with_threshold = (target - predicted).abs() <= self.threshold

        self.true_positives += (predicted & target).sum()
        self.false_positives += (predicted & ~target).sum()


class RMSE(torchutil.metrics.RMSE):
    """Root mean square error of pitch distance in cents"""
    def update(self, predicted, target):
        predicted += + 1e-8
        pred_rmse = ppn.OCTAVE * torch.log2(predicted)
        # pred_rmse[predicted == ppn.NO_PITCH_BIN] = 0

        target += + 1e-8
        targ_rmse = ppn.OCTAVE * torch.log2(target)
        # targ_rmse[target == ppn.NO_PITCH_BIN] = 0
        super().update(
            pred_rmse,
            targ_rmse)


class RPA(torchutil.metrics.Average):
    """Raw prediction accuracy"""
    def __init__(self, threshold: int) -> None:
        super().__init__()
        self.threshold = threshold

    def update(self, predicted, target):
        difference = ppn.tools.misc.cents(predicted, target)
        super().update(
            (torch.abs(difference) < self.threshold).sum(),
            predicted.numel())
