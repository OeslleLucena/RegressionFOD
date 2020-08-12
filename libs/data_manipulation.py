import os
import numpy as np
import nibabel as nib


class DataManipulation:
    @staticmethod
    def new_dir(out_path):
        path_list = out_path.split('/')[-1]
        path_list = path_list.split('.')
        if len(path_list) > 1:
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
        else:
            if not os.path.exists(out_path):
                os.makedirs(out_path)

    @staticmethod
    def niftynet_format(img):
        # Adding time axis for NiftyNet and saving normalized data
        if len(img.shape) == 4:
            return img[..., np.newaxis, :]
        else:
            return img[..., np.newaxis]

    @staticmethod
    def nibabel_format(img):
        if len(img.shape) == 5:
            return img[..., 0, :]
        else:
            return img[..., 0]

    @staticmethod
    def read_nibabel_img(path):
        img = nib.load(path)
        affine = img.affine
        return img.get_data(), affine

    @staticmethod
    def save_nibabel_img(img, affine, out):
        img = nib.Nifti1Image(img, affine)
        nib.save(img, out)

    @staticmethod
    def convert_nibabel(fod_niftynet_path, fod_nibabel_path):
        DataManipulation().new_dir(os.path.dirname(fod_nibabel_path))
        img, affine = DataManipulation().read_nibabel_img(fod_niftynet_path)
        img = DataManipulation().nibabel_format(img)
        DataManipulation().save_nibabel_img(img, affine, fod_nibabel_path)




