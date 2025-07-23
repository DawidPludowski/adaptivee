from argparse import ArgumentParser, Namespace


def get_download_args() -> Namespace:
    parser = ArgumentParser(description="Download datasets for training.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="resources/liltab/raw",
        help="Path to save datasets.",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=-1,
        required=False,
        help="Maximum number of datasets to use from the source.",
    )
    return parser.parse_args()


def get_split_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--inference-train-frac", type=float)
    parser.add_argument("--encoder-train-frac", type=float)
    parser.add_argument("--inference-frac", type=float)
    parser.add_argument("--seed", type=int, default=123, required=False)
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--inference-outer-split", action="store_true")
    parser.add_argument("--encoder-outer-split", action="store_true")
    return parser.parse_args()


def get_pretraining_weights_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model-list-id", type=str)
    parser.add_argument("--input-path", type=str)
    parser.add_argument(
        "--alpha", type=float, help="Parameters to softmax weighting"
    )
    parser.add_argument(
        "--use-onehot",
        action="store_true",
        help="Change Softmax weighter to onehot; overwrite alpha",
    )
    return parser.parse_args()
