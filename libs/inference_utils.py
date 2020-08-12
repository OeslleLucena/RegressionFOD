import numpy as np
from ast import literal_eval
import torch
from torch.utils.data import DataLoader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator

import sys
sys.path.extend('../')

from libs import dataset
from libs.highresnet import HighRes3DNet
from libs.unet import Unet3D
from libs import niftynet_utils
from libs import model_utils


TRAINING = 'training'
VALIDATION = 'validation'
INFERENCE = 'inference'



class Regression:
    def __init__(self, data_param, grouping_param, data_split_file,
                 patch_size, windows_per_image, window_border,
                 queue_length, num_input_channels, num_output_channels,
                 num_dilations, num_highresnet_blocks, activation,
                 padding_mode, network, gpu_used, batch_size,
                 num_workers, checkpoint_path, inference_path):

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

        self.device = self.get_device(gpu_used)
        self.dataloaders = self.get_dataloaders(batch_size, num_workers)
        self.iteration = 0
        self.best_val_loss = None
        self.best_val_epoch = None
        self.validation_every_n_epochs = 10
        self.iteration = 0
        self.inference = self.run_inference(window_border,
                                            inference_path,
                                            checkpoint_path)

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
                                                             data_augmentation=True,
                                                             whitening=False)

        readers[VALIDATION] = niftynet_utils.add_preprocessing(readers[VALIDATION],
                                                               VALIDATION,
                                                               data_augmentation=False,
                                                               whitening=False)

        readers[INFERENCE] = niftynet_utils.add_preprocessing(readers[INFERENCE],
                                                              INFERENCE,
                                                              whitening=False,
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

        if network == 'unet':
            model = Unet3D(num_input_channels=num_input_channels,
                           num_output_channels=num_output_channels,
                           activation=act)

        elif network == 'highresnet':
            model = HighRes3DNet(in_channels=num_input_channels,
                                 out_channels=num_output_channels,
                                 num_dilations=num_dilations,
                                 num_highresnet_blocks=num_highresnet_blocks,
                                 activation=act,
                                 padding_mode=padding_mode)

        else:
            print('xx', network)
            raise Exception('Invalid phase choice: {}'.format(
                {'network': ['highresnet', 'unet']}))

        return model


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


    def save_weights(self, model, checkpoint_path):
        torch.save(model.state_dict(), checkpoint_path)

    def load_weights(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


    def run_inference(self, window_border, inference_path, checkpoint_path):

        output = GridSamplesAggregator(image_reader=self.samplers[INFERENCE].reader,
                                       window_border=window_border,
                                       interp_order=3,
                                       output_path=inference_path)

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
