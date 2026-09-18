# %%
import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path(__file__).resolve().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from plotly_utils import line


# %%
# ReLu
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.max(t.zeros_like(x), x)


tests.test_relu(ReLU)

# %%
# Linear


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 1. create a tensor shape (out_features, in_features)
        # 2. populate every value as a uniform distribution between the interval
        upper = 1 / np.sqrt(in_features)
        lower = -1 / np.sqrt(in_features)
        weight = (upper - lower) * t.rand(out_features, in_features) + lower

        self.weight = nn.Parameter(weight)
        if bias is True:
            bias = (upper - lower) * t.rand(out_features) + lower
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """

        if self.bias is not None:
            return x @ self.weight.T + self.bias

        return x @ self.weight.T

    def extra_repr(self) -> str:
        return f"Geoff and Dan's Linear module with {self.in_features} in_features and {self.out_features} out_features with bias={self.bias is not None}"


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)

linear = Linear(512, 80, True)
print(linear.extra_repr())


# %%
# Flatten
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])


flatten = Flatten(start_dim=1, end_dim=1)
upper = 1 / np.sqrt(512)
lower = -1 / np.sqrt(512)
weight = (upper - lower) * t.rand(512, 80) + lower
print(weight.shape)
print(flatten.forward(weight).shape)

# %%
# MLP


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28 * 28, 100)
        self.linear2 = Linear(100, 10)
        self.flatten = Flatten()
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        flattened = self.flatten.forward(x)
        linear1_result = self.linear1.forward(flattened)
        relu_result = self.relu.forward(linear1_result)
        linear2_result = self.linear2.forward(relu_result)
        return linear2_result


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)

# %%

MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
    ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(
        exercises_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM
    )
    mnist_testset = datasets.MNIST(
        exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM
    )

    # # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Get the first batch of test data, by starting to iterate over `mnist_testloader`
for img_batch, label_batch in mnist_testloader:
    print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
    break

# Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
for img, label in mnist_testset:
    print(f"{img.shape=}\n{label=}\n")
    break

t.testing.assert_close(img, img_batch[0])
assert label == label_batch[0].item()

# %%
device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# If this is CPU, we recommend figuring out how to get cuda access (or MPS if you're on a Mac).
print(device)

# %%
# Training loop

model = SimpleMLP().to(device)

batch_size = 128
epochs = 3

mnist_trainset, _ = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []

for epoch in range(epochs):
    pbar = tqdm(mnist_trainloader)

    for imgs, labels in pbar:
        # Move data to device, perform forward pass
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)

        # Calculate loss, perform backward pass
        loss = F.cross_entropy(logits, labels)
        loss.backward()  # this part calculates the loss/gradients
        optimizer.step()  # this part updates the model params
        optimizer.zero_grad()  # this part re-zeroes the gradients

        # Update logs & progress bar
        loss_list.append(loss.item())
        pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}")

# %%

line(
    loss_list,
    x_max=epochs * len(mnist_trainset),
    labels={"x": "Examples seen", "y": "Cross entropy loss"},
    title="SimpleMLP training on MNIST",
    width=700,
)

# %%


@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g.
    self.batch_size=64. Any of these fields can also be overridden when you create an instance, e.g.
    SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3


# %%


def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], list[float], SimpleMLP]:
    """
    Trains the model, using training parameters from the `args` object.

    Returns:
        The model, and lists of loss & accuracy.
    """

    # 1. train
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []

    for epoch in range(args.epochs):
        pbar = tqdm(mnist_trainloader)

        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.3f}")

    # validate

    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size)
    accuracy_list = []

    # Validation loop
    num_correct_classifications = 0
    for imgs, labels in mnist_testloader:
        # Move data to device, perform forward pass in inference mode
        imgs, labels = imgs.to(device), labels.to(device)
        with t.inference_mode():
            logits = model(imgs)

        # Compute num correct by comparing argmaxed logits to true labels
        predictions = t.argmax(logits, dim=1)
        num_correct_classifications += (predictions == labels).sum().item()

    # Compute & log total accuracy
    accuracy = num_correct_classifications / len(mnist_testset)
    accuracy_list.append(accuracy)

    # pbar_test = tqdm(mnist_testloader)
    # for imgs, labels in pbar_test:
    #     imgs, labels = imgs.to(device), labels.to(device)
    #     with t.inference_mode():
    #         logits = model(imgs)  # logits -> score of which number it is; 1 - 10

    #     probabilities = t.softmax(logits, dim=1)
    #     guesses = t.max(probabilities, dim=1)

    #     accuracy = t.eq(labels, guesses.indices).int().sum() / (labels.size())[0]
    #     accuracy_list.append(accuracy.item())

    return loss_list, accuracy_list, model


args = SimpleMLPTrainingArgs()
loss_list, accuracy_list, model = train(args)

line(
    y=[loss_list, [0.1] + accuracy_list],  # we start by assuming a uniform accuracy of 10%
    use_secondary_yaxis=True,
    x_max=args.epochs * len(mnist_trainset),
    labels={"x": "Num examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
    title="SimpleMLP training on MNIST",
    width=800,
)


# %%
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.

        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # YOUR CODE HERE - define & initialize `self.weight`
        n_in = in_channels * kernel_size * kernel_size
        upper = 1 / np.sqrt(n_in)
        lower = -1 / np.sqrt(n_in)
        weight = (upper - lower) * t.rand(
            out_channels, in_channels, kernel_size, kernel_size
        ) + lower

        self.weight = nn.Parameter(weight)
        # print(weight.shape)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")


# %%
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d."""
        return F.max_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join(
            [f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]]
        )


# %%
class BatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """

        if self.training:
            mean = t.mean(x, dim=(0, 2, 3))
            var = t.var(x, dim=(0, 2, 3), unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        # Rearrange
        reshape = lambda x: einops.rearrange(x, "channels -> 1 channels 1 1")

        x_normed = (x - reshape(mean)) / (reshape(var) + self.eps).sqrt()
        x_affine = x_normed * reshape(self.weight) + reshape(self.bias)
        return x_affine

    def extra_repr(self) -> str:
        return f"{self.num_batches_tracked}"


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)


# %%
class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return x.mean(dim=(2, 3))


tests.test_averagepool(AveragePool)

# %%
# Building ResNet


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a
        `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right
        branch. Declare it second using another `Sequential`.
        """
        super().__init__()
        is_shape_preserving = (first_stride == 1) and (
            in_feats == out_feats
        )  # determines if right branch is identity
        self.is_shape_preserving = is_shape_preserving

        self.kernel_size = 3
        self.padding = 1
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        self.relu = ReLU()

        self.conv2d_1 = Conv2d(
            self.in_feats, self.out_feats, self.kernel_size, self.first_stride, self.padding
        )
        self.batchnorm2d_1 = BatchNorm2d(self.out_feats)
        self.conv2d_2 = Conv2d(
            self.out_feats, self.out_feats, self.kernel_size, stride=1, padding=self.padding
        )
        self.batchnorm2d_2 = BatchNorm2d(self.out_feats)

        if is_shape_preserving is not True:
            self.conv2d_optional = Conv2d(
                self.in_feats, self.out_feats, kernel_size=1, stride=self.first_stride, padding=0
            )
            self.batchnorm2d_optional = BatchNorm2d(self.out_feats)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass. If no downsampling block is present, the addition should just add
        the left branch's output to the input.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)
        """

        strided_conv = self.conv2d_1.forward(x)
        norm_1 = self.batchnorm2d_1.forward(strided_conv)
        relu_1 = self.relu.forward(norm_1)
        conv = self.conv2d_2.forward(relu_1)
        norm_2 = self.batchnorm2d_2.forward(conv)

        if self.is_shape_preserving is not True:
            optional_conv = self.conv2d_optional.forward(x)
            optional_norm = self.batchnorm2d_optional.forward(optional_conv)
            result = norm_2 + optional_norm
        else:
            result = norm_2 + x

        return self.relu.forward(result)


tests.test_residual_block(ResidualBlock)


# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """
        An n_blocks-long sequence of ResidualBlock where only the first block uses the provided
        stride.
        """
        super().__init__()
        self.result = nn.Sequential(
            ResidualBlock(in_feats, out_feats, first_stride=first_stride),
            *[ResidualBlock(out_feats, out_feats) for _ in range(1, n_blocks)],
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.result(x)


tests.test_block_group(BlockGroup)


# %%
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        out_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        block_groups = []
        prev_out_feats = out_feats0
        for n, out_feats, first_stride in zip(
            n_blocks_per_group, out_features_per_group, first_strides_per_group
        ):
            block_groups.append(BlockGroup(n, prev_out_feats, out_feats, first_stride))
            prev_out_feats = out_feats

        self.result = nn.Sequential(
            Conv2d(3, out_feats0, 7, 2, 3),
            BatchNorm2d(out_feats0),
            ReLU(),
            MaxPool2d(3, 2),
            *block_groups,
            AveragePool(),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        return self.result.forward(x)


my_resnet = ResNet34()

# (1) Test via helper function `print_param_count`
target_resnet = (
    models.resnet34()
)  # without supplying a `weights` argument, we just initialize with random weights
utils.print_param_count(my_resnet, target_resnet)

# (2) Test via `torchinfo.summary`
print("My model:", torchinfo.summary(my_resnet, input_size=(1, 3, 64, 64)), sep="\n")
print(
    "\nReference model:",
    torchinfo.summary(target_resnet, input_size=(1, 3, 64, 64), depth=2),
    sep="\n",
)

# %%


def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    """Copy over the weights of `pretrained_resnet` to your resnet."""

    # Get the state dictionaries for each model, check they have the same number of parameters &
    # buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the
    # pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(
            mydict.items(), pretraineddict.items()
        )
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
my_resnet = copy_weights(my_resnet, pretrained_resnet).to(device)
print("Weights copied successfully!")

# %%

IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = section_dir / "resnet_inputs"

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]

# %%

display(images[0])

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0).to(device)
assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)

# %%
