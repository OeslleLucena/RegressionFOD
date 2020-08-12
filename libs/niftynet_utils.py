from niftynet.engine.sampler_weighted_v2 import WeightedSampler
from niftynet.io.image_reader import ImageReader
from niftynet.engine.signal import TRAIN, VALID, INFER
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.layer.pad import PadLayer

from libs.fod_rand_rotation import FODRandomRotationLayer as Rotate


def get_reader(data_param, grouping_param, image_sets_partitioner,
               phase):
    # Using Nifty Reader
    if phase == 'training':
        image_reader = ImageReader().initialise(
            data_param, grouping_param,
            file_list=image_sets_partitioner.get_file_list(TRAIN))

    elif phase == 'validation':
        image_reader = ImageReader().initialise(
            data_param, grouping_param,
            file_list=image_sets_partitioner.get_file_list(VALID))

    elif phase == 'inference':
        # TODO: need to improve that
        if data_param['mask']:
            del data_param['mask']
        if grouping_param['sampler']:
            del grouping_param['sampler']
        image_reader = ImageReader().initialise(
            data_param, grouping_param,
            file_list=image_sets_partitioner.get_file_list(INFER))

    else:
        raise Exception('Invalid phase choice: {}'.format(
            {'phase': ['training', 'validation', 'inference']}))
    return image_reader


# TODO: rewrite this function
def add_preprocessing(image_reader, phase, window_border=None,
                      data_augmentation=False):

    rotation_layer = Rotate()
    rotation_layer.init_uniform_angle([-25.0, 25.0])

    if phase == 'training':
        if data_augmentation:
            image_reader.add_preprocessing_layers([rotation_layer])

    elif phase == 'inference':
        if window_border:
            pad_layer = PadLayer(image_name=('image', 'label'),
                                 border=window_border,
                                 mode='constant')
            image_reader.add_preprocessing_layers([pad_layer])
        else:
            raise Exception('Invalid window border!')
    else:
        raise Exception('Invalid phase choice: {}'.format(
            {'phase': ['training', 'validation', 'inference']}))
    return image_reader


def get_sampler(image_reader, patch_size, phase,
                windows_per_image=None, window_border=None):
    if phase in ('training', 'validation'):
        if windows_per_image:
            sampler = WeightedSampler(image_reader,
                                      window_sizes=patch_size,
                                      windows_per_image=windows_per_image)
        else:
            raise Exception('Invalid windows per image!')

    elif phase == 'inference':
        if window_border:
            sampler = GridSampler(image_reader,
                                  window_sizes=patch_size,
                                  window_border=window_border)
        else:
            raise Exception('Invalid window border!')

    else:
        raise Exception('Invalid phase choice: {}'.format(
            {'phase': ['training', 'validation', 'inference']}))
    return sampler
