import torch


def anomaly_correlation_coefficient(model, dataloader, device_id):
    # surface: B, variable, x
    # upper: B, variable, height, x

    initialized = False
    logit_surface_squared = None
    target_surface_squared = None
    logit_upper_squared = None
    target_upper_squared = None
    nominator_surface = None
    nominator_upper = None

    dims = [0, -1]
    for idx, batch in enumerate(dataloader):
        batch = {k: v.to(device_id) for k, v in batch.items()}
        output = model(batch)
        output = {k: v.detach() for k, v in output.items()}
        if not initialized:
            initialized = True
            logit_surface_squared = (output["logits_surface"] ** 2).sum(dim=dims)
            target_surface_squared = (batch["target_surface"] ** 2).sum(dim=dims)
            logit_upper_squared = (output["logits_upper"] ** 2).sum(dim=dims)
            target_upper_squared = (batch["target_upper"] ** 2).sum(dim=dims)

            nominator_surface = (
                output["logits_surface"] * batch["target_surface"]
            ).sum(dim=dims)
            nominator_upper = (output["logits_upper"] * batch["target_upper"]).sum(
                dim=dims
            )
        else:
            logit_surface_squared += (output["logits_surface"] ** 2).sum(dim=dims)
            target_surface_squared += (batch["target_surface"] ** 2).sum(dim=dims)
            logit_upper_squared += (output["logits_upper"] ** 2).sum(dim=dims)
            target_upper_squared += (batch["target_upper"] ** 2).sum(dim=dims)

            nominator_surface += (
                output["logits_surface"] * batch["target_surface"]
            ).sum(dim=dims)
            nominator_upper += (output["logits_upper"] * batch["target_upper"]).sum(
                dim=dims
            )
        if idx > 2:
            break

    denominator_surface = torch.sqrt(logit_surface_squared * target_surface_squared)
    denominator_upper = torch.sqrt(logit_upper_squared * target_upper_squared)

    acc_surface = nominator_surface / denominator_surface
    acc_upper = nominator_upper / denominator_upper

    return dict(acc_surface=acc_surface, acc_upper=acc_upper)


def rmse(model, dataloader, device_id):
    # surface: B, variable, x
    # upper: B, variable, height, x

    initialized = False
    rmse_surface = None
    rmse_upper = None

    dims = [0, -1]
    n_batches = 0
    for idx, batch in enumerate(dataloader):
        batch = {k: v.to(device_id) for k, v in batch.items()}
        output = model(batch)
        output = {k: v.detach() for k, v in output.items()}
        n_pixels = batch["target_surface"].shape[-1]
        n_samples = batch["target_surface"].shape[0]
        if not initialized:
            initialized = True
            rmse_surface_batches = (
                torch.sqrt(
                    ((output["logits_surface"] - batch["target_surface"]) ** 2).sum(
                        dim=-1
                    )
                )
                / n_pixels
            )
            rmse_upper_batches = (
                torch.sqrt(
                    ((output["logits_upper"] - batch["target_upper"]) ** 2).sum(dim=-1)
                )
                / n_pixels
            )
            rmse_surface = rmse_surface_batches.sum(dim=0) / n_samples
            rmse_upper = rmse_upper_batches.sum(dim=0) / n_samples
        else:
            rmse_surface_batches = (
                torch.sqrt(
                    ((output["logits_surface"] - batch["target_surface"]) ** 2).sum(
                        dim=-1
                    )
                )
                / n_pixels
            )
            rmse_upper_batches = (
                torch.sqrt(
                    ((output["logits_upper"] - batch["target_upper"]) ** 2).sum(dim=-1)
                )
                / n_pixels
            )
            rmse_surface += rmse_surface_batches.sum(dim=0) / n_samples
            rmse_upper += rmse_upper_batches.sum(dim=0) / n_samples
        n_batches += 1
        if idx > 2:
            break

    rmse_surface /= n_batches
    rmse_upper /= n_batches
    return dict(surface=rmse_surface, upper=rmse_upper)
