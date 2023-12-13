#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path

# import onnxruntime as ort

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import create_metric


# from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig
from experiments.weather.models.swin_hp_pangu import SwinHPPanguConfig
from experiments.weather.models.swin_hp_pangu import SwinHPPangu

# from lib.models.mlp import MLPConfig
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import symlink_checkpoint_files
from lib.files import prepare_results

from lib.data_factory import get_factory as get_dataset_factory
from lib.model_factory import get_factory as get_model_factory

from lib.render_psql import add_artifact

from experiments.weather.data import DataHP
from experiments.weather.data import DataHPConfig

NSIDE = 64
EPOCHS = 10


def create_config(ensemble_id):
    loss = torch.nn.L1Loss()

    def reg_loss(output, batch):
        # breakpoint()
        # breakpoint()
        # return loss(output["logits_upper"], batch["target_upper"])  # + 0.25 * loss(

        return loss(output["logits_upper"], batch["target_upper"]) + 0.25 * loss(
            output["logits_surface"], batch["target_surface"]
        )
        # return loss(output["logits_surface"], batch["target_surface"])

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=SwinHPPanguConfig(
            base_pix=12,
            nside=NSIDE,
            dev_mode=False,
            depths=[2, 6, 6, 2],
            # num_heads=[6, 12, 12, 6],
            num_heads=[8, 16, 16, 8],
            embed_dims=[192 // 2, 384 // 2, 384 // 2, 192 // 2],
            # embed_dims=[16, 384 // 16, 384 // 16, 192 // 16],
            # embed_dims=[x for x in [16, 32, 32, 16]],
            window_size=[2, 16],  # int(32 * (NSIDE / 256)),
            use_cos_attn=False,
            use_v2_norm_placement=True,
            drop_rate=0,  # ,0.1,
            attn_drop_rate=0,  # ,0.1,
            drop_path_rate=0,
            rel_pos_bias="earth",
            # shift_size=8,  # int(16 * (NSIDE / 256)),
            shift_size=8,  # int(16 * (NSIDE / 256)),
            shift_strategy="nest_roll",
            ape=False,
            patch_size=4,  # int(16 * (NSIDE / 256)),
        ),
        train_data_config=DataHPConfig(nside=NSIDE),
        val_data_config=None,  # DataHPConfig(nside=NSIDE),
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
            # kwargs=dict(weight_decay=3e-6, lr=5e-3),
        ),
        batch_size=1,
        ensemble_id=ensemble_id,
        _version=50,
    )
    train_eval = TrainEval(
        train_metrics=[create_metric(reg_loss)], validation_metrics=[]
    )  # create_regression_metrics(torch.nn.functional.l1_loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=3),
        train_config=train_config,
        train_eval=train_eval,
        epochs=300,
        save_nth_epoch=1,
        validate_nth_epoch=20,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()

    data_factory = get_dataset_factory()
    data_factory.register(DataHPConfig, DataHP)
    model_factory = get_model_factory()
    model_factory.register(SwinHPPanguConfig, SwinHPPangu)

    ensemble_config = create_ensemble_config(create_config, 1)
    ensemble = create_ensemble(ensemble_config, device_id)

    ds = data_factory.create(DataHPConfig(nside=NSIDE))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    result_path = prepare_results(
        Path(__file__).parent,
        f"{Path(__file__).stem}_{ensemble_config.members[0].train_config.model_config.__class__.__name__}_nside_{NSIDE}",
        ensemble_config,
    )
    symlink_checkpoint_files(ensemble, result_path)

    # options = ort.SessionOptions()
    # options.enable_cpu_mem_arena = False
    # options.enable_mem_pattern = False
    # options.enable_mem_reuse = False
    # options.intra_op_num_threads = 16

    # cuda_provider_options = {
    # "arena_extend_strategy": "kSameAsRequested",
    # }

    # ort_session_3 = ort.InferenceSession(
    #     "experiments/weather/pangu_models/pangu_weather_3.onnx",
    #     sess_options=options,
    #     providers=[("CUDAExecutionProvider", cuda_provider_options)],
    # )
    def save_and_register(name, array):
        path = result_path / f"{name}.npy"

        np.save(
            path,
            array.detach().cpu().numpy(),
        )
        add_artifact(ensemble_config.members[0], name, path)

    # for batch in tqdm.tqdm(dl):
    #     batch = {k: v.to(device_id) for k, v in batch.items()}

    #     start = time.time()
    #     output = ensemble.members[0](batch)
    #     model_time = time.time()
    #     print(f"Model time: {model_time - start}, Sample {batch['sample_id']}")
    # save_and_register("of_surface", output["logits_surface"])
    # save_and_register("if_surface", batch["input_surface"])
    # save_and_register("tf_surface", batch["target_surface"])
    # save_and_register("of_upper", output["logits_upper"])
    # save_and_register("if_upper", batch["input_upper"])
    # save_and_register("tf_upper", batch["target_upper"])

    # save_and_register("of_surface.npy", output["logits_surface"])
    # np.save(
    #     result_path / "if_surface.npy",
    #     batch["input_surface"].detach().cpu().numpy(),
    # )
    # np.save(
    #     result_path / "tf_surface.npy",
    #     batch["target_surface"].detach().cpu().numpy(),
    # )

    # dh, dh_target = ds.get_driscoll_healy(ids[0])
    # te5s = ds.get_template_e5s()
    # pangu_output_upper, pangu_output_surface = ort_session_3.run(
    #     None,
    #     dict(
    #         input=dh.upper.to_array().to_numpy(),
    #         input_surface=dh.surface.to_array().to_numpy(),
    #     ),
    # )
    # pangu_surface_xds = numpy_to_xds(pangu_output_surface, te5s.surface)
    # pangu_upper_xds = numpy_to_xds(pangu_output_upper, te5s.upper)
    # pangu_np_surface, pangu_np_upper = ds.e5_to_numpy(
    #     cdstest.ERA5Sample(surface=pangu_surface_xds, upper=pangu_upper_xds)
    # )
    # np.save(result_path / "pangu_pred_surface.npy", pangu_np_surface)
    # np.save(result_path / "pangu_pred_upper.npy", pangu_np_upper)
    # np.save(
    #     result_path / "pangu_pred_surface.npy",
    #     batch["input_surface"].detach().cpu().numpy()[0],
    # )
    # np.save(
    #     result_path / "pangu_pred_upper.npy",
    #     batch["input_upper"].detach().cpu().numpy()[0],
    # )
    # break
