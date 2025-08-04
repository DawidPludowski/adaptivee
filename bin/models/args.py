from argparse import ArgumentParser, Namespace


def get_des_baselines_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=False)
    parser.add_argument(
        "--save-out",
        action="store_true",
        help="save predictions to .npx files",
    )
    parser.add_argument("--model-list-id", type=str)

    return parser.parse_args()


def get_pretrain_encoder_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--val-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, default="results")
    parser.add_argument("--model-list-id", type=str, required=True)
    parser.add_argument("--alpha-name", type=str, default="alpha=0.1")
    parser.add_argument("--n-hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-size", type=int, default=30)

    return parser.parse_args()
