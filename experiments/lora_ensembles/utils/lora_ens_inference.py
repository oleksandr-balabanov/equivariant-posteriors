from typing import Dict, List
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer
from lib.serialization import DeserializeConfig, get_checkpoint_path, deserialize_model
from lib.train_dataclasses import TrainRun
from lib.paths import get_model_epoch_checkpoint_path


@dataclass
class DeserializedModelStateDict:
    state_dict: Dict[str, torch.Tensor]
    epoch: int


def deserialize_model_state_dict(config: DeserializeConfig):
    train_config = config.train_run.train_config
    checkpoint_path = get_checkpoint_path(train_config)
    if (
        not (checkpoint_path / "model").is_file()
        or not (checkpoint_path / "epoch").is_file()
    ):
        return None
    else:
        print(f"{checkpoint_path}")

    try:
        model_state_dict = torch.load(
            checkpoint_path / "model", map_location=torch.device(config.device_id)
        )

        model_epoch = torch.load(checkpoint_path / "epoch")

        if model_epoch != config.train_run.epochs:
            model_epoch_checkpoint = get_model_epoch_checkpoint_path(
                config.train_run.train_config, config.train_run.epochs
            )
            if not (model_epoch_checkpoint).is_file():
                raise Exception(
                    f"The requested epoch ({config.train_run.epochs}) is not available in {model_epoch_checkpoint}."
                )
            model_state_dict = torch.load(
                model_epoch_checkpoint, map_location=torch.device(config.device_id)
            )
            print(
                f"Loaded earlier epoch {config.train_run.epochs}, the latest epoch is {model_epoch}."
            )

    except Exception as e:
        print(f"Failed to deserialize_model: {e}")
        return None

    return DeserializedModelStateDict(state_dict=model_state_dict, epoch=model_epoch)


@dataclass
class LORAEnsemble:
    model: torch.nn.Module
    ensemble_state_dicts: List[Dict[str, torch.Tensor]]

    def ensemble_forward(self, batch):
        outputs = []
        for member_state_dict in self.ensemble_state_dicts:
            self.model.load_state_dict(member_state_dict)
            output = self.model(batch)
            output = {k: v.detach() for k, v in output.items()}
            outputs.append(output)
        return outputs
    
    def load_member(self, member_id):
            member_state_dict = self.ensemble_state_dicts[member_id]
            self.model.load_state_dict(member_state_dict)

    def member_forward(self, batch):
        output = self.model(batch)
        output = {k: v.detach() for k, v in output.items()}
        return output

    def member_generate(self, **kwargs):
        return self.model.model.generate(**kwargs)




def create_lora_ensemble(member_configs: List[TrainRun], device_id, checkpoint_epochs = None):
    state_dicts = []
    for member_config in member_configs:
        if checkpoint_epochs:
            member_config.epochs = checkpoint_epochs
        if checkpoint_epochs == 0:
            member_config.epochs = 1 # to load the state dic
            
        deserialized_state_dict = deserialize_model_state_dict(
            DeserializeConfig(train_run=member_config, device_id=device_id)
        )
        print("Here", deserialized_state_dict.epoch, member_config.epochs)
        if not deserialized_state_dict.epoch >= member_config.epochs:
            print(
                f"WARNING: Member not fully trained ({deserialized_state_dict.epoch}/{member_config.epochs} epochs)"
            )

        if checkpoint_epochs != 0: 
            state_dicts.append(deserialized_state_dict.state_dict)
        else:
            trivial_state_dic = zero_out_state_dict(deserialized_state_dict.state_dict)
            state_dicts.append(trivial_state_dic)

    deserialized_model = deserialize_model(
        DeserializeConfig(train_run=member_configs[0], device_id=device_id)
    )
    return LORAEnsemble(
        model=deserialized_model.model, ensemble_state_dicts=state_dicts
    )

def zero_out_state_dict(state_dict):
    for key in state_dict:
        state_dict[key] = torch.zeros_like(state_dict[key])

    return state_dict

