from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class Args:
    '''
    a Args class that maintain all arguments for model training
    '''
    data_path: str
    output_path: str
    max_seq_length: int = 128
    train_batch_size_per_device: int = 16
    train_epochs: int = 1
    accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    # available: "linear","linearconstant","constant"
    lr_scheduler: str = "linear"
    training_lr: float = 5e-5

    seed: int = 1
    weight_decay: float = 1
    override: bool = True
    eda_aug: bool = False
    eval_batch_size_per_device: int = 32
    eval_steps: int = -1
    # batch size for data tokenization
    tok_bs: int = 1000
    # the weight between categories loss and priority loss
    alpha: float = 0.5
    base_model_path_or_name: str = "bert-base-uncased"

