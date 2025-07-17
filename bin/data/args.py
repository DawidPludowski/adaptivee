from argparse import ArgumentParser, Namespace


def get_download_args() -> Namespace:
    parser = ArgumentParser(description="Download datasets for training.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="resources/liltab",
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
    pass


def get_pretraining_weights_args() -> Namespace:
    pass
