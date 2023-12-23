# -*- coding: utf-8 -*-


from typing import Union, Callable, Dict, Tuple
import pytorch_lightning as ptl
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_3d: bool = False,
        use_normalization: bool = True,
    ):
        """
        A recurring structure throughout the UNet architecture.
        For clarity's sake, we define it once, and reuse the block.

        :param in_features: How many features do we start with?
        :param out_features: How many features do we need as output?
        :param is_3d: Is this block part of a 3D UNet? If not, will use 2D layers
        :param use_normalization: Whether to use a normalization layer between the
                                  convolution and ReLU layers
        """
        super().__init__()

        if is_3d:
            layers = [
                nn.Conv3d(in_features, out_features, kernel_size=3, padding=1, bias=False),
            ]

            if use_normalization:
                layers.append(nn.BatchNorm3d(out_features))

            layers += [
                nn.ReLU(inplace=True),
                nn.Conv3d(out_features, out_features, kernel_size=3, padding=1, bias=False),
            ]

            if use_normalization:
                layers.append(nn.BatchNorm3d(out_features))

            layers.append(nn.ReLU(inplace=True))
            
        else:
            layers = [
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=False),
            ]

            if use_normalization:
                layers.append(nn.BatchNorm2d(out_features))

            layers += [
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False),
            ]

            if use_normalization:
                layers.append(nn.BatchNorm2d(out_features))

            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through this model

        :param x: input data
        :return: The processed input data
        """
        return self.net(x)


class DownBlock(ptl.LightningModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_3d: bool = False,
        use_normalization: bool = True,
    ):
        """
        Defines an Encoder block for a UNet architecture

        :param in_features: How many features do we start with?
        :param out_features: How many features do we need as output?
        :param is_3d: Is this block part of a 3D UNet? If not, will use 2D layers
        :param use_normalization: Whether to use a normalization layer between the
                                  convolution and ReLU layers
        """

        super().__init__()

        if is_3d:
            self.net = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                DoubleConv(in_features, out_features, is_3d, use_normalization),
            )
        else:
            self.net = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(in_features, out_features, is_3d, use_normalization),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through this model

        :param x: input data
        :return: The processed input data
        """

        return self.net(x)


class UpBlock(ptl.LightningModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_3d: bool = False,
        use_transpose: bool = False,
        use_normalization: bool = True,
    ):
        """
        Defines a Decoder block for a UNet architecture

        :param in_features: How many features do we start with?
        :param out_features: How many features do we need as output?
        :param is_3d: Is this block part of a 3D UNet? If not, will use 2D layers
        :param use_transpose: How do we upscale our data?
                              If true, will use a transposed convolution
                              If False, will use a bi-/triliniar upscale followed
                              by a convolution
        :param use_normalization: Whether to use a normalization layer between the
                                  convolution and ReLU layers
        """

        super().__init__()

        if use_transpose:
            if is_3d:
                self.upsample = nn.ConvTranspose3d(
                    in_features, in_features // 2, kernel_size=2, stride=2
                )
            else:
                self.upsample = nn.ConvTranspose2d(
                    in_features, in_features // 2, kernel_size=2, stride=2
                )
        else:
            if is_3d:
                self.upsample = nn.Sequential(
                    nn.Upsample(
                        scale_factor=2,
                        mode="trilinear" if is_3d else "bilinear",
                        align_corners=True,
                    ),
                    nn.Conv3d(in_features, in_features // 2, kernel_size=1),
                )
            else:
                self.upsample = nn.Sequential(
                    nn.Upsample(
                        scale_factor=2,
                        mode="trilinear" if is_3d else "bilinear",
                        align_corners=True,
                    ),
                    nn.Conv2d(in_features, in_features // 2, kernel_size=1),
                )

        self.conv = DoubleConv(
            in_features=in_features,
            out_features=out_features,
            is_3d=is_3d,
            use_normalization=use_normalization,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through this model

        :param x: Input data
        :param skip: The input from the skip connection
        :return: The processed input data
        """
        x = self.upsample(x)

        # Here is where we would pad if we needed to.

        # Concatenate along the channels axis
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
    
class UNet(ptl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        loss_function: Union[nn.Module, Callable, None],
        num_layers: int = 5,
        input_features: int = 16,
        is_3d: bool = False,
        use_transpose: bool = True,
        use_normalization: bool = True,
        lr: float = 1e-3,
        final_activation: Union[nn.Module, Callable] = None,
    ):
        """
        The UNet architecture, as described in the article by Ronneberger et al.
        This implementation has been adapted from the PyTorch Lightning Bolts package.

        :param input_channels: Number of channels in the input image
        :param num_classes: Number of classes in the output
        :param num_layers: The depth of the model.
        :param loss_function: The loss function to use during training, validation and testing
        :param input_features: How many features should the input layer have?
        :param is_3d: Do we need to use 3D layers instead of 2D?
        :param use_transpose: Whether to use transposed convolutions instead of bi/trilinear
                              upsamples in the expanding path
         :param use_normalization: Whether to use normalization layers between the convolutions
                                  ReLU layers.
        :param lr: The learning rate
        :param final_activation: What layer should we use as final activation?
                                 Not used in forward(), but in the training/validation/testing
                                 step functions.
        """
        super().__init__()

        # The following calls are not mandatory for a working LightningModule, but they
        # give us some nice-to-haves for our own benefit.

        # Saving hyperparameters allows us to track these values in our Tensorboard logger
        self.save_hyperparameters(ignore=["loss_function", "final_activation"])

        # This is used later for our ModelSummary(), giving it an example input so that
        # the Callback can run it through the model and keep track of its shape.
        self.example_input_array = torch.rand((1, input_channels, 240, 240))

        # Back to regular model building!
        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        self.num_layers = num_layers
        self.lr = lr
        self.final_activation = final_activation
        self.loss_function = loss_function

        # We start our model architecture with an input DoubleConv
        layers = [DoubleConv(input_channels, input_features, is_3d, use_normalization)]

        # Down path
        feats = input_features
        for _ in range(num_layers - 1):
            layers.append(DownBlock(feats, feats * 2, is_3d, use_normalization))
            feats *= 2

        # Up path
        for _ in range(num_layers - 1):
            layers.append(UpBlock(feats, feats // 2, is_3d, use_transpose, use_normalization))
            feats //= 2

        # Our final convolution
        if is_3d:
            layers.append(nn.Conv3d(feats, num_classes, kernel_size=1))
        else:
            layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through this model

        :param x: Input data
        :return: The processed input data
        """
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        # Here we define our optimizer, and any learning rate scheduling
        # Adam usually works well, so we just use that
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # As for the scheduler, we reduce our learning rate whenever our validation
                # loss doesn't decrease for 5 epochs in a row.
                # Other schedulers exist, but using one is not mandatory, you'll just have a
                # fixed learning rate during the course of training.
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.1,
                    patience=5,
                    verbose=True,
                ),
                "monitor": "val_loss",
            },
        }

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        # This is the same type of code that you'd usually find in your
        # original PyTorch training for loop.
        x, y = batch
        y_hat = self.forward(x)
        if self.final_activation is not None:
            y_hat = self.final_activation(y_hat)

        loss = self.loss_function(y_hat, y)
        output = {"loss": loss}

        # self.log() and self.log_dict() are more used for making sure
        # that the current progress is displayed properly, and written
        # by the TensorboardLogger. the use of either is not mandatory
        self.log_dict(output, prog_bar=True, on_epoch=True)

        # training_step() has to return either:
        #   - A scalar of the loss
        #   - A dictionary, with the key "loss" present
        # this return is used by Lightning internally
        return output

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        # This is also where we could log images to Tensorboard
        # to visually track our progress.

        x, y = batch
        y_hat = self.forward(x)
        if self.final_activation is not None:
            y_hat = self.final_activation(y_hat)

        loss = self.loss_function(y_hat, y)
        output = {"val_loss": loss}
        self.log_dict(output, prog_bar=True, on_epoch=True)

        # validation_step() does _not_ require a return,
        # in fact, you don't even need to use the same
        # loss function as you do in training_step().

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        # Quite similar to, if not the same as, the validation_step()
        x, y = batch
        y_hat = self.forward(x)
        if self.final_activation is not None:
            y_hat = self.final_activation(y_hat)

        loss = self.loss_function(y_hat, y)
        output = {"test_loss": loss}
        self.log_dict(output)
