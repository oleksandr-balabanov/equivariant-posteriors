import torch
from pathlib import Path
import pandas as pd

from lib.uncertainty import uncertainty
from lib.files import prepare_results


def uq_for_ensemble(dl, ensemble, ensemble_config, model_name: str, device_id):
    uq = uncertainty(dl, ensemble, device_id)

    def save_uq(config, uq, filename):
        result_path = prepare_results(Path(__file__).parent, model_name, config)
        data = torch.concat(
            [
                uq.MI[:, None].cpu(),
                uq.H[:, None].cpu(),
                uq.sample_ids.cpu(),
                uq.mean_pred[:, None].cpu(),
                uq.targets[:, None].cpu(),
                torch.where(
                    uq.targets[:, None].cpu() == uq.mean_pred[:, None].cpu(),
                    1.0,
                    0.0,
                ),
            ],
            dim=-1,
        )
        df = pd.DataFrame(
            columns=["MI", "H"] + uq.sample_id_spec + ["pred", "target", "accuracy"],
            data=data.numpy(),
        )

        df.to_csv(result_path / filename)

    save_uq(ensemble_config, uq, f"{model_name}_uq_{dl.dataset.name()}.csv")
