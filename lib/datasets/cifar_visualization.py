import torch
import torchvision as tv
from lib.train_dataclasses import TrainEpochState


class UnNormalize(tv.transforms.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def visualize_cifar(plt, state: TrainEpochState, device_id):
    state.model.eval()
    input, target, ids = state.train_dataloader.dataset[
        0
    ]  # = next(iter(state.val_dataloader))
    input = torch.tensor(input).unsqueeze(0)
    un = UnNormalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    input = un(input)

    # with torch.no_grad():
    # input = input.to(device_id, non_blocking=True)
    # output = state.model(input)
    # digits = torch.argmax(output["predictions"].cpu(), dim=-1)

    images = input.cpu().reshape(-1, 3, 32, 32)
    images = 255.0 * images[:1]
    images = tv.transforms.functional.resize(images, [32, 64])
    image_grid = tv.utils.make_grid(images.long(), 1)
    image_grid = image_grid.numpy().transpose((1, 2, 0))[:, :, 0].tolist()

    # digits = digits[:4].cpu().split(2)
    # digits_str = "\n".join([str(x.tolist()) for x in digits])
    plt.plotsize(64, 32)
    plt.matrix_plot(image_grid)
    # plt.text(digits_str, 0, 2, color="red")
