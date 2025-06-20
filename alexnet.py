from torch import nn
from torchvision import models

from .cnn_base import CNNModelBase


class AlexNetModel(CNNModelBase):
    """AlexNet classifier using ``torchvision.models.alexnet``."""

    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = False,
        lr: float = 1e-3,
        epochs: int = 1,
    ) -> None:
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        model = models.alexnet(weights=weights)
        if num_classes != 1000:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        super().__init__(model, lr=lr, epochs=epochs)
