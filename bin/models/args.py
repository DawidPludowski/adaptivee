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
