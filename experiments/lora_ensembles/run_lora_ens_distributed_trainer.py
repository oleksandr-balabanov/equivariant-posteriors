import lib.distributed_trainer
import experiments.lora_ensembles.run_lora_ens_train as run_lora_ens_train

if __name__ == "__main__":
    run_lora_ens_train.register_model_and_dataset()
    lib.distributed_trainer.distributed_train()
