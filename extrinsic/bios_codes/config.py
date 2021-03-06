import enum
import argparse

import dataclasses


@dataclasses.dataclass
class TrainConfig:
    model: str = 'counts'  # counts | han | rnn
    hidden_size: int = 64
    attention_size: int = 64
    dropout: float = 0.25
    trainable_embeddings: bool = False

    scrubbed: bool = False
    use_class_weight: bool = True

    max_grad_norm: float = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    nb_epochs: int = 30
    batch_size: int = 512
    early_stopping_patience: int = 5
    min_epochs: int = 5


@dataclasses.dataclass
class PreprocessConfig:
    counts_max_df: float = 0.8  # 0.9
    counts_min_df: int = 50  # 20
    counts_binary: bool = True

    seq_max_len: int = 110
    names_pca_n: int = 0



def parse_args(config_class, description):
    parser = argparse.ArgumentParser(description=description, argument_default=argparse.SUPPRESS)

    for field in dataclasses.fields(config_class):
        choices = None
        default = None

        if issubclass(field.type, enum.Enum):
            choices = [m.value for m in field.type]
            default = field.default.value

        if field.type in [int, float, bool, str]:
            default = field.default

        parser.add_argument(
            f'--{field.name}', type=field.type, choices=choices, help=f'{field.type.__name__}, default: {default}'
        )

    provided_args = parser.parse_args()
    provided_args = vars(provided_args)

    cfg = config_class(**provided_args)

    return cfg
