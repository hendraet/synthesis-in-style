from typing import Type

from networks.doc_ufcn.doc_ufcn import DocUFCN, DocUFCNNoDropout, PixelShuffleDocUFCN


def get_doc_ufcn(version: str) -> Type[DocUFCN]:
    if version == 'base':
        segmentation_network_class = DocUFCN
    elif version == 'no_dropout':
        segmentation_network_class = DocUFCNNoDropout
    elif version == 'pixelshuffle':
        segmentation_network_class = PixelShuffleDocUFCN
    else:
        raise NotImplementedError(f"the network you wish for is not implemented, you wished for {version}")
    return segmentation_network_class
