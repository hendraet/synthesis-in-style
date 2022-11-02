from lightning_modules.ema_net_lightning import EmaNetSegmenter
from lightning_modules.docUFCN_lightning import DocUFCNSegmenter
from lightning_modules.trans_u_net_lightning import TransUNetSegmenter


def get_segmenter_class(config):
    if config['network'] == 'DocUFCN':
        segmenter_class = DocUFCNSegmenter
    elif config['network'] == 'TransUNet':
        segmenter_class = TransUNetSegmenter
    elif config['network'] == 'EMANet':
        segmenter_class = EmaNetSegmenter
    else:
        raise NotImplementedError
    return segmenter_class
