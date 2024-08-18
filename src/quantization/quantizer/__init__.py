import copy

from .lsq import LsqQuantizer
from .quantizer import IdentityQuantizer
from .twn import TwnQuantizer
from .hbc import hbcQuantizer
from .pact import PACTQuantizer


def build_quantizer(cfg):
    if cfg is None:
        return IdentityQuantizer()

    cfg = copy.deepcopy(cfg)
    if cfg['mode'] == "Identity":
        quant = IdentityQuantizer
    elif cfg['mode'] == "LSQ":
        quant = LsqQuantizer
    elif cfg['mode'] == "TWN":
        quant = TwnQuantizer
    elif cfg['mode'] == "HBC":
        quant = hbcQuantizer
    elif cfg['mode'] == "PACT":
        quant = PACTQuantizer
    else:
        raise NotImplementedError

    cfg.pop('mode')
    return quant(**cfg)
