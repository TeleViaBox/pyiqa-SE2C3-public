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
        
        # Build the network using the architecture defined in se2c3_arch.py
        self.net = build_network(opt['network'])
        
        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt['train']['lr'])

        # Initialize the loss function
        self.criterion = build_loss(opt['train']['criterion'])
        
        # Initialize learning rate scheduler if specified
        if 'lr_scheduler' in opt['train']:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=opt['train']['lr_scheduler']['step_size'], 
                gamma=opt['train']['lr_scheduler']['gamma']
            )

        # Setup logging
        self.logger = get_root_logger()

    def init_training_settings(self):
        super(SE2C3Model, self).init_training_settings()

    def setup_optimizers(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt['train']['lr'])
        if 'lr_scheduler' in self.opt['train']:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.opt['train']['lr_scheduler']['step_size'], 
                gamma=self.opt['train']['lr_scheduler']['gamma']
            )

    def feed_data(self, data):
        self.img_in = data['img_in'].to(self.device)
        self.img_gt = data['img_gt'].to(self.device)

    def net_forward(self):
        self.output = self.net(self.img_in)

    def optimize_parameters(self, current_iter):
        self.net.train()
        self.optimizer.zero_grad()
        self.net_forward()
        
        # Calculate loss
        loss = self.criterion(self.output, self.img_gt)
        loss.backward()
        self.optimizer.step()
        
        # Optionally step the scheduler
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        
        # Log the loss
        if current_iter % self.opt['logger']['print_freq'] == 0:
            self.logger.info(f'Iteration {current_iter}: Loss {loss.item()}')

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.net_forward()
            # Evaluate using metrics
            results = calculate_metric(self.output, self.img_gt, self.opt['metrics'])
            self.logger.info(f'Test Metrics: {results}')
            return results

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.net.eval()
        total_loss = 0.0
        num_samples = len(dataloader)
        
        with torch.no_grad():
            for data in dataloader:
                self.feed_data(data)
                self.net_forward()
                
                # Calculate loss
                loss = self.criterion(self.output, self.img_gt)
                total_loss += loss.item()
                
                # Save images if required
                if save_img:
                    visuals = self.get_current_visuals()
                    img_path = osp.join(self.opt['path']['visualization'], f"{current_iter}_{data['img_path']}.png")
                    imwrite(tensor2img(visuals['output']), img_path)
            
            avg_loss = total_loss / num_samples
            self.logger.info(f'Validation Loss: {avg_loss}')
            return avg_loss

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        return self.dist_validation(dataloader, current_iter, tb_logger, save_img)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        self.logger.info(f"Validation at iteration {current_iter} on dataset {dataset_name}.")

    def save(self, epoch, current_iter, save_net_label='net'):
        save_filename = f'{save_net_label}_{epoch}_{current_iter}.pth'
        save_path = osp.join(self.opt['path']['checkpoints'], save_filename)
        torch.save(self.net.state_dict(), save_path)
        self.logger.info(f'Saved model at {save_path}')

    def load(self, epoch, current_iter, load_net_label='net'):
        load_filename = f'{load_net_label}_{epoch}_{current_iter}.pth'
        load_path = osp.join(self.opt['path']['checkpoints'], load_filename)
        self.net.load_state_dict(torch.load(load_path, map_location=self.device))
        self.logger.info(f'Loaded model from {load_path}')
