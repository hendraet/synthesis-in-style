from stylegan_code_finder.training_builder.doc_ufcn_train_builder import DocUFCNTrainBuilder
from stylegan_code_finder.training_builder.ema_net_train_builder import EMANetTrainBuilder
from stylegan_code_finder.training_builder.pixel_ensemble_train_builder import PixelEnsembleTrainBuilder
from stylegan_code_finder.training_builder.trans_u_net_train_builder import TransUNetTrainBuilder


def get_train_builder_class(config):
    if config['network'] == 'DocUFCN':
        train_builder_class = DocUFCNTrainBuilder
    elif config['network'] == 'TransUNet':
        train_builder_class = TransUNetTrainBuilder
    elif config['network'] == 'EMANet':
        train_builder_class = EMANetTrainBuilder
    elif config['network'] == 'PixelEnsemble':
        train_builder_class = PixelEnsembleTrainBuilder
    else:
        raise NotImplementedError
    return train_builder_class
