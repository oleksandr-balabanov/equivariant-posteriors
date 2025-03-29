import torch
from lib.train_dataclasses import TrainEpochState


def visualize_spiral(plt, state: TrainEpochState, device_id):
    state.model.eval()
    outputs = []
    inputs = []
    with torch.no_grad():
        for batch in state.val_dataloader:
            batch = {k: v.to(device_id) for k, v in batch.items()}

            output = state.model(batch)["predictions"].cpu()
            outputs.append(output.detach().cpu())
            inputs.append(batch["input"].detach().cpu())

    input = torch.concat(inputs, dim=0)
    output = torch.concat(outputs, dim=0)
    class1 = input[output[:, 0] <= 0.5].cpu()
    class2 = input[output[:, 0] > 0.5].cpu()
    for data in [class1, class2]:
        x = data[:, 0].reshape(-1).numpy().tolist()
        y = data[:, 1].reshape(-1).numpy().tolist()
        plt.scatter(x, y)
