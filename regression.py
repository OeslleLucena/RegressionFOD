import os
import numpy as np
import time
from ast import literal_eval
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
import argparse
import sys
sys.path.extend('../')

from libs import dataset
from libs.highresnet import HighRes3DNet
from libs.unet import Unet3D
from libs import loss
from libs import niftynet_utils
from libs import model_utils
from libs.data_manipulation import DataManipulation


TRAINING = 'training'
VALIDATION = 'validation'
INFERENCE = 'inference'
np.random.seed(0)


class Regression:
    def __init__(self, seed, data_param, grouping_param, data_split_file,
                 patch_size, windows_per_image, window_border,
                 queue_length, num_input_channels, num_output_channels,
                 num_dilations, num_highresnet_blocks, activation,
                 padding_mode, network, optimizer_name, learning_rate,
                 momentum, weight_decay, loss_name, gpu_used, batch_size,
                 num_workers, learning_rate_mode, num_epochs, checkpoint_path,
                 inference_path, tensorboadx_path):
        self.setup_reproducibility(seed)
        self.data_param, \
            self.grouping_param, self.image_sets_partitioner = self.read_data(data_param,
                                                                              grouping_param,
                                                                              data_split_file)
        self.readers = self.get_readers(window_border)
        self.samplers = self.get_samplers(patch_size,
                                          windows_per_image,
                                          window_border)
        self.datasets = self.get_datasets(queue_length)
        self.model = self.get_model(num_input_channels,
                                    num_output_channels,
                                    num_dilations,
                                    num_highresnet_blocks,
                                    activation,
                                    padding_mode,
                                    network)
        self.optimizer = self.get_optimizer(optimizer_name,
                                            learning_rate,
                                            momentum,
                                            weight_decay)
        self.criterion = self.get_loss_function(loss_name)
        self.device = self.get_device(gpu_used)
        self.dataloaders = self.get_dataloaders(batch_size, num_workers)
        self.scheduler = self.get_scheduler(learning_rate_mode)
        self.iteration = 0
        self.best_val_loss = None
        self.best_val_epoch = None
        self.validation_every_n_epochs = 10
        self.iteration = 0
        self.tensorboadx_path = tensorboadx_path
        self.train = self.run_training(num_epochs,
                                       checkpoint_path)
        self.inference = self.run_inference(window_border,
                                            inference_path,
                                            checkpoint_path)

#
    def setup_reproducibility(self, seed):
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def read_data(self, data_param, grouping_param, data_split_file):
        # Dictionary with parameters for NiftyNet Reader
        data_param = literal_eval(data_param)
        grouping_param = literal_eval(grouping_param)
        image_sets_partitioner = ImageSetsPartitioner().initialise(
            data_param=data_param,
            data_split_file=data_split_file,
            new_partition=False
        )
        return data_param, grouping_param, image_sets_partitioner

    def get_readers(self, window_border):
        readers = {x: niftynet_utils.get_reader(self.data_param,
                                                self.grouping_param,
                                                self.image_sets_partitioner,
                                                x)
                   for x in [TRAINING, VALIDATION, INFERENCE]}
        # adding preprocessing layers
        readers[TRAINING] = niftynet_utils.add_preprocessing(readers[TRAINING],
                                                             TRAINING,
                                                             data_augmentation=True)

        readers[INFERENCE] = niftynet_utils.add_preprocessing(readers[INFERENCE],
                                                              INFERENCE,
                                                              window_border=window_border)
        return readers

    def get_samplers(self, patch_size, windows_per_image, window_border):
        samplers = {x: niftynet_utils.get_sampler(self.readers[x],
                                                  tuple(patch_size),
                                                  x,
                                                  windows_per_image)
                    for x in [TRAINING, VALIDATION]}

        samplers[INFERENCE] = niftynet_utils.get_sampler(self.readers[INFERENCE],
                                                         tuple(patch_size),
                                                         INFERENCE,
                                                         window_border=window_border)
        return samplers

    def get_datasets(self, queue_length):
        datasets = {x: dataset.PatchBasedDataset(queue_length=queue_length,
                                                 reader=self.readers[x],
                                                 sampler=self.samplers[x])
                    for x in [TRAINING, VALIDATION]}
        return datasets

    def get_model(self, num_input_channels, num_output_channels,
                  num_dilations, num_highresnet_blocks, activation,
                  padding_mode, network):

        act = model_utils.activation_function(activation)

        if network == 'highresnet':
            model = HighRes3DNet(in_channels=num_input_channels,
                                 out_channels=num_output_channels,
                                 num_dilations=num_dilations,
                                 num_highresnet_blocks=num_highresnet_blocks,
                                 activation=act,
                                 padding_mode=padding_mode)

        if network == 'unet':
            model = Unet3D(num_input_channels=num_input_channels,
                           num_output_channels=num_output_channels,
                           activation=act)

        else:
            raise Exception('Invalid phase choice: {}'.format(
                {'network': ['highresnet', 'unet']}))

        return model

    def get_optimizer(self, optimizer_name, learning_rate,
                      momentum, weight_decay):
        parameters = self.model.parameters()
        optimizer = getattr(optim, optimizer_name)(
            parameters, lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay)

        return optimizer

    def get_loss_function(self, loss_name):
        if loss_name == 'l2':
            criterion = loss.L2Loss()
        elif loss_name == 'l2_nn':
            criterion = loss.L2Loss_niftynet()
        else:
            raise Exception('Invalid phase choice: {}'.format(
                {'loss': ['mse', 'nrmse']}))
        return criterion

    def get_device(self, gpu_used):
        if torch.cuda.is_available():
            print('[INFO] GPU available.')
            device = torch.device("cuda:{}".format(gpu_used)
                                  if torch.cuda.is_available() else "cpu")
        else:
            raise Exception(
                "[INFO] No GPU found or Wrong gpu id, please run without --cuda")
        return device

    def get_dataloaders(self, batch_size, num_workers):
        dataloaders = {x: DataLoader(self.datasets[x],
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     drop_last=True)
                       for x in [TRAINING, VALIDATION]}
        return dataloaders

    def get_scheduler(self, learning_rate_mode):
        # TODO: write the other schedulers
        if learning_rate_mode == 'step':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                  step_size=50,
                                                  gamma=0.5)
        else:
            raise Exception('Invalid scheduler choice: {}'.format(
                {'schedulers': ['step', 'cyclical']}))
        return scheduler


    def save_weights(self, model, checkpoint_path):
        torch.save(model.state_dict(), checkpoint_path)

    def load_weights(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


    def run_one_epoch(self, model, phase):
        if phase == TRAINING:
            model.train()
        elif phase == VALIDATION:
            model.eval()
        else:
            raise Exception('Wrong model phase: {}'.format(
                {'model phases': [TRAINING, VALIDATION]}))

        running_loss = []  # epoch running loss
        iterable = tqdm(self.dataloaders[phase])
        for inputs, labels in iterable:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            # forward + backpropagation for training stage only
            if phase == TRAINING:
                # set gradients True only in the training phase
                with torch.set_grad_enabled(phase == TRAINING):
                    outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.iteration += 1
                running_loss.append(loss.mean().item())
                self.tbxWriter.add_scalar('data/train_iteration_loss',
                                          loss.data,
                                          self.iteration)

            # forward only
            if phase == VALIDATION:
                with torch.no_grad():
                    outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.mean().item())
        return running_loss

    def run_training(self, num_epochs, checkpoint_path):
        self.tbxWriter = SummaryWriter(self.tensorboadx_path)
        DataManipulation().new_dir(checkpoint_path)
        model = self.model.to(self.device)
        since = time.time()
        for epoch in range(num_epochs):
            self.scheduler.step()
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']

            print('[INFO] Epoch {}/{} lr {}'.format(epoch + 1, num_epochs, lr))
            # training
            running_loss = self.run_one_epoch(model, TRAINING)
            train_epoch_loss = np.asarray(running_loss).mean()
            self.tbxWriter.add_scalar('data/train_epoch_loss',
                                      train_epoch_loss,
                                      epoch + 1)
            print('[INFO] training epoch loss: {:.4f}'.format(train_epoch_loss))
            if epoch == 0:
                self.best_val_loss = train_epoch_loss
                self.best_val_epoch = epoch + 1
                torch.save(model.state_dict(), checkpoint_path.format(epoch + 1))

            # validation
            if (epoch + 1) % self.validation_every_n_epochs == 0:
                running_loss = self.run_one_epoch(model, VALIDATION)
                val_epoch_loss = np.asarray(running_loss).mean()
                self.tbxWriter.add_scalar('data/validation epoch loss',
                                          val_epoch_loss,
                                          epoch + 1)
                print('[INFO] validation epoch loss: {:.4f}'.format(val_epoch_loss))

                # save model based os validation
                if val_epoch_loss < self.best_val_loss:
                    self.best_val_loss = val_epoch_loss
                    self.best_val_epoch = epoch + 1
                    torch.save(model.state_dict(), checkpoint_path)
                    print('[INFO] Checkpoint {} saved!'.format(self.best_val_epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                            time_elapsed % 60))
        print('Best validation loss {:4f} at epoch {}'.format(self.best_val_loss,
                                                              self.best_val_epoch))

    def run_inference(self, window_border, inference_path, checkpoint_path):

        output = GridSamplesAggregator(image_reader=self.samplers[INFERENCE].reader,
                                       window_border=window_border,
                                       interp_order=3,
                                       output_path=inference_path)

        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        for batch_output in self.samplers[INFERENCE]():
            window = batch_output['image']
            # [...,0,:] eliminates time coordinate from NiftyNet Volume
            window = window[..., 0, :]
            window = np.transpose(window, (0, 4, 1, 2, 3))
            window = torch.Tensor(window).to(self.device)
            with torch.no_grad():
                outputs = self.model(window)
            outputs = outputs.cpu().numpy()
            outputs = np.transpose(outputs, (0, 2, 3, 4, 1))
            output.decode_batch(outputs,
                                batch_output['image_location'])

def parsing_data():
    parser = argparse.ArgumentParser(description='CSD regression')
    parser.add_argument('-dataset',
                        default='HCP',
                        type=str, help='dataset to be used')
    parser.add_argument('-shell', default=1000,
                        type=int, help='shell to be used')
    parser.add_argument('-fold', default=1,
                        type=int, help='fold number')
    parser.add_argument('-gpu_used', default=0,
                        type=int, help='gpu card to be used')

    opt = parser.parse_args()

    return opt


def main():
    opt = parsing_data()
    num_workers = 0
    num_input_channels = 15
    num_output_channels = 15
    windows_per_image = 40
    patch_size = (32,32,32)
    window_border = (8,8,8)
    batch_size = 40
    loss_name = 'l2_nn'
    optimizer_name = 'RMSprop'
    learning_rate = 3e-2
    weight_decay = 1e-6
    queue_length = 5*batch_size
    momentum = 0
    num_epochs = 1
    activation = 'prelu'
    learning_rate_mode = 'step'
    network = 'unet'
    padding_mode = 'reflect'
    num_dilations = 2
    num_highresnet_blocks = 2
    seed = 0
    fold = opt.fold
    gpu_used = opt.gpu_used
    shell = opt.shell
    dataset = opt.dataset
    experiment_dir = f'/home/ol18/Exps/Model_Fitting/Journal/cnn_results_2/{dataset}/{shell}'
    data_csv_dir = f'/home/ol18/Exps/Model_Fitting/Journal/{dataset}_data/csv'


    print(f'{fold}, {gpu_used}, {dataset}, {shell}, {network}')


    data_param = "{'CSD': " \
                 "{'path_to_search': " \
                 f"'/home/ol18/Exps/Model_Fitting/Journal/{dataset}_data/single_shell_2t_csf_lmax4_norm_nn/{shell}'," \
                 "'filename_contains': 'wmfod'," \
                 "'interp_order': '3'}," \
                 "'mask': " \
                 "{'path_to_search': " \
                 f"'/home/ol18/Exps/Model_Fitting/Journal/{dataset}_data/masks_nn'," \
                 "'filename_contains': 'mask'," \
                 "'interp_order': '0'}," \
                 "'label': " \
                 "{'path_to_search': " \
                 f"'/home/ol18/Exps/Model_Fitting/Journal/{dataset}_data/multi_shell_lmax4_wm_norm_nn'," \
                 "'filename_contains': 'wmfod'," \
                 "'interp_order': '3'}}"


    grouping_param = "{'image': ('CSD',), 'sampler':('mask',), 'label': ('label',)}"

    experiment_name = 'regression_csd_rot_mtnorm_' + network + '_num_dilations_'\
                      + str(num_dilations) \
                      + '_num_highresnet_blocks_' + str(num_highresnet_blocks) \
                      + '_windows_per_image_' + str(windows_per_image) \
                      + '_loss_' + loss_name + '_' + activation \
                      + '_bsize_' + str(batch_size) + '_psize_'\
                      + str(patch_size[0]) + '_epochs_' + str(num_epochs) \
                      + '_in_' + str(num_input_channels) + '_out_' \
                      + str(num_output_channels) + '_lr_' + str(learning_rate) \
                      + '_wd_' + str(weight_decay) + '_momentum_' \
                      + str(momentum) + '_lr_mode_' + learning_rate_mode \
                      + '_opt_' + optimizer_name

    data_split_file = os.path.join(data_csv_dir,
                                   f'fold_reg{fold}_split.csv')
    checkpoint_path = os.path.join(experiment_dir, experiment_name,
                                  'cps', f'fold_{fold}.pth')
    inference_path = os.path.join(experiment_dir, experiment_name,
                                  'preds', f'fold_{fold}')
    tensorboadx_path = os.path.join(experiment_dir, experiment_name,
                                  'tensorboardX_files', f'fold_{fold}')


    Regression(seed, data_param, grouping_param, data_split_file,
               patch_size, windows_per_image, window_border,
               queue_length, num_input_channels, num_output_channels,
               num_dilations, num_highresnet_blocks, activation,
               padding_mode, network, optimizer_name, learning_rate,
               momentum, weight_decay, loss_name, gpu_used, batch_size,
               num_workers, learning_rate_mode, num_epochs, checkpoint_path,
               inference_path, tensorboadx_path)

if __name__ == '__main__':
    main()