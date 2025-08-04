from argparse import ArgumentParser, Namespace


def get_run_experiment_args() -> Namespace:
    parser = ArgumentParser(description="Download datasets for training.")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--encoder-path", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--model-list-id", type=str)
    parser.add_argument("--alpha", type=str)
    parser.add_argument("--reweighter", type=str)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument(
        "--use-autogluon",
        action="store_true",
    )
    return parser.parse_args()
