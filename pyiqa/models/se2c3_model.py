import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from pyiqa.archs import build_network
from pyiqa.losses import build_loss
from pyiqa.metrics import calculate_metric
from pyiqa.utils import get_root_logger, imwrite, tensor2img
from pyiqa.utils.registry import MODEL_REGISTRY
from .general_iqa_model import GeneralIQAModel


@MODEL_REGISTRY.register()
class SE2C3Model(GeneralIQAModel):
    """SE2C3 model for IQA."""

    def __init__(self, opt):
        super(SE2C3Model, self).__init__(opt)

    def init_training_settings(self):
        super(SE2C3Model, self).init_training_settings()

    def setup_optimizers(self):
        super(SE2C3Model, self).setup_optimizers()

    def feed_data(self, data):
        super(SE2C3Model, self).feed_data(data)

    def net_forward(self, net):
        return super(SE2C3Model, self).net_forward(net)

    def optimize_parameters(self, current_iter):
        super(SE2C3Model, self).optimize_parameters(current_iter)

    def test(self):
        super(SE2C3Model, self).test()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        super(SE2C3Model, self).dist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        super(SE2C3Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        super(SE2C3Model, self)._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def save(self, epoch, current_iter, save_net_label='net'):
        super(SE2C3Model, self).save(epoch, current_iter, save_net_label)