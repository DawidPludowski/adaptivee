from adaptivee.encoders import LiltabEncoder
from adaptivee.reweighting import (
    SimpleReweight,
    DirectionReweight,
    DirectionConstantReweight,
    OneTakesAllReweight,
)
from adaptivee.target_weights import SoftMaxWeighter, OneHotWeighter
from functools import partial


def load_encoder(path: str):
    return partial(LiltabEncoder, model_path=path)


def get_reweighter(name: str):
    if name == "simple":
        return SimpleReweight
    elif name == "one":
        return OneTakesAllReweight

    class_, param = name.split("-")
    if class_ == "direction":
        return partial(DirectionReweight, step_size=float(param))
    elif class_ == "direction_constant":
        return partial(DirectionConstantReweight, step_size=float(param))
    else:
        raise ValueError


def get_target_weighter(alpha: float | str):
    if alpha == "inf":
        return OneHotWeighter
    else:
        return partial(SoftMaxWeighter, alpha=float(alpha))
