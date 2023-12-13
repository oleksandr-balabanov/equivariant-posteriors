import lib.distributed_trainer

import experiments.lora_ensembles.lora_ensemble as lora_ensemble

if __name__ == "__main__":
    lora_ensemble.register_model_and_dataset()
    lib.distributed_trainer.distributed_train()
