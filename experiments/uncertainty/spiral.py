import torch
import os
import pandas as pd
from pathlib import Path

from lib.data_factory import get_factory
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import symlink_checkpoint_files
from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.classification_metrics import create_classification_metrics
from lib.data_registry import DataSpiralsConfig
from lib.data_registry import DataUniformConfig
from lib.data_registry import DataSubsetConfig
from lib.datasets.spiral_visualization import visualize_spiral
from lib.models.mlp import MLPClassConfig
import lib.uncertainty as uncertainty
from lib.files import prepare_results
from lib.render_psql import add_ensemble_artifact


def create_config(ensemble_id):
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss(output, batch):
        return ce_loss(output["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=MLPClassConfig(widths=[100, 100]),
        # model_config=TransformerConfig(
        #     embed_d=20, mlp_dim=10, n_seq=2, batch_size=500, num_layers=2, num_heads=1
        # ),
        train_data_config=DataSpiralsConfig(seed=0, N=1000),
        val_data_config=DataSpiralsConfig(seed=1, N=500),
        loss=loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(lr=0.01, weight_decay=0.00001)
        ),
        batch_size=500,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_spiral, 2)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=100,
        save_nth_epoch=20,
        validate_nth_epoch=20,
    )
    return train_run


def main():
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"
    ensemble_config = create_ensemble_config(create_config, 10)
    ensemble = create_ensemble(ensemble_config, device_id)

    result_path = prepare_results(
        Path(__file__).parent, Path(__file__).stem, ensemble_config
    )
    symlink_checkpoint_files(ensemble, result_path)

    data_factory = get_factory()
    # ds = data_factory.create(DataSpiralsConfig(seed=5, N=1000))
    ds = data_factory.create(DataUniformConfig(min=-3, max=3, N=500))
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=500,
        shuffle=False,
        drop_last=True,
    )
    uncertainties = uncertainty.uncertainty(dataloader, ensemble, device_id)
    r = torch.linalg.vector_norm(ds.uniform.xs.squeeze(), dim=-1)
    data = torch.concat(
        [
            uncertainties.MI[:, None].cpu(),
            uncertainties.H[:, None].cpu(),
            uncertainties.sample_ids[:, None].cpu(),
            r[:, None],
            ds.uniform.xs.squeeze(),
            uncertainties.mean_pred[:, None].cpu(),
        ],
        dim=-1,
    )
    df = pd.DataFrame(
        columns=["MI", "H", "id", "r", "x", "y", "pred"], data=data.numpy()
    )
    df.to_csv(result_path / "uncertainty_mlp.csv")
    add_ensemble_artifact(ensemble_config, "uq", result_path / "uncertainty_mlp.csv")


if __name__ == "__main__":
    main()
