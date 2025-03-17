from lib.ensemble import create_ensemble_config
from lib.ensemble import request_ensemble
from lib.files import prepare_results
from lib.distributed_trainer import distributed_train
from experiments.lora_ensembles.train.lora_ens_train_config import create_lora_ens_train_run_config
import argparse
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import LoraEnsTrainConfig

def create_lora_ens_train_run_config_factory(ens_train_config: LoraEnsTrainConfig):
    def create_config(ensemble_id: int):
        return create_lora_ens_train_run_config(ensemble_id, ens_train_config)
    return create_config

def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--member_id', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--lora_l2', type=float, required=True)
    parser.add_argument('--lora_dropout', type=float, required=True)
    parser.add_argument('--regular_l2', type=float, required=True)
    parser.add_argument('--use_generative_next_token_loss', type=str, required=True)
    parser.add_argument('--max_len_train', type=int, required=True)
    parser.add_argument('--max_len_val', type=int, required=True)
    args = parser.parse_args()

    if args.use_generative_next_token_loss.lower() == 'true':
        use_generative_next_token_loss = True
    elif args.use_generative_next_token_loss.lower() == 'false':
        use_generative_next_token_loss = False
    else:
        raise ValueError("Invalid value for is_generative_next_token_loss. Use 'true' or 'false'.")

    # train config
    ens_train_config = LoraEnsTrainConfig(
        train_dataset=args.train_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_dropout=args.lora_dropout,
        lora_l2=args.lora_l2,
        regular_l2=args.regular_l2,
        max_len_train=args.max_len_train,
        max_len_val=args.max_len_val,
        use_generative_next_token_loss=use_generative_next_token_loss
    )

    # create ensemble_config
    create_new_lora_ens_train_run_config = create_lora_ens_train_run_config_factory(ens_train_config)
    ensemble_config = create_ensemble_config(create_new_lora_ens_train_run_config, args.member_id+1)
    prepare_results("lora_ensemble", ensemble_config.members)
    print("ensemble_config finished")
    request_ensemble(ensemble_config)

    # train one member
    distributed_train([ensemble_config.members[-1]])
    print("ensemble finished")

if __name__ == "__main__":
    main()
