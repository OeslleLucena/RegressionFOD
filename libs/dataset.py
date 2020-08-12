from random import shuffle
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import multiprocessing as mp


class PatchBasedDataset(Dataset):
    def __init__(
            self,
            queue_length,
            reader,
            sampler):

        self.reader = reader
        self.sampler = sampler
        self.samples_per_volume = self.sampler.window.n_samples
        self.queue = Queue(self.sampler, queue_length)
        self.volume_idx = np.random.randint(self.reader.num_subjects)

    def __len__(self):
        """
        We define the number of iterations per epoch as
        the number of subjects time the samples per volume
        """
        return self.sampler.n_subjects * self.samples_per_volume

    def __getitem__(self, idx):
        """
        Note that parameter idx is ignored
        """
        if not self.queue:
            # self.volume_idx = self.queue.fill(self.volume_idx)
            self.volume_idx = self.queue.fill()

        samples = self.queue.pop() # popping last sample with size == batch_size?
        return samples['image'], samples['label']

class Queue(list):
    def __init__(self, sampler, queue_length, percentage_cores=0, show_progress=True):
        super().__init__()
        self.sampler = sampler
        self.reader = sampler.reader
        self.samples_per_volume = sampler.window.n_samples
        self.queue_length = queue_length
        self.current_volume_idx = np.random.randint(self.reader.num_subjects)
        self.percentage_cores = percentage_cores
        self.show_progress = show_progress

        if self.percentage_cores > 0:
            self.fill = self.fill_parallel
        else:
            self.fill = self.fill_single

    def fill_single(self):
        iterations = self.queue_length // self.samples_per_volume
        progress = trange(iterations)
        for _ in progress:
            progress.set_description(f'Filling queue ({self.current_volume_idx})')
            samples = self.sample_from_sampler(self.current_volume_idx)
            self.extend(samples)
            self.current_volume_idx += 1
            if self.current_volume_idx == self.sampler.n_subjects:
                self.current_volume_idx = 0
        shuffle(self)

    def fill_parallel(self):
        iterations = self.queue_length // self.samples_per_volume
        num_cpus = mp.cpu_count()
        num_usable_cpus = int(num_cpus * self.percentage_cores / 100)
        subject_indices = []

        # Get next subjects
        for i in range(iterations):
            subject_indices.append(self.current_volume_idx)
            self.current_volume_idx += 1
            if self.current_volume_idx == self.sampler.n_subjects:
                self.current_volume_idx = 0

        # https://stackoverflow.com/a/45276885/3956024
        with mp.Pool(num_usable_cpus) as pool:
            imap = pool.imap(self.sample_from_sampler, subject_indices)
            samples_groups = (
                tqdm(imap, total=iterations, leave=False)
                if self.show_progress else imap
            )
            samples_groups = list(samples_groups)
        samples = [
            sample
            for sample_group in samples_groups
            for sample in sample_group
        ]
        self.extend(samples)
        shuffle(self)

    def sample_from_sampler(self, volume_idx):
        samples = []
        samples_dict = self.sampler(idx=volume_idx)
        for i in range(self.samples_per_volume):
            image = samples_dict['image'][i]
            label = samples_dict['label'][i]

            # convert to PyTorch tensor
            image = niftynet_to_pytorch_format(image)
            label = niftynet_to_pytorch_format(label)

            image = torch.from_numpy(image.copy()).float()
            label = torch.from_numpy(label.copy()).float()

            location = samples_dict['image_location'][i]
            subject_index = location[0]
            subject_id = self.reader.get_subject_id(subject_index)
            voxel_ini = torch.tensor(location[1:4], dtype=torch.int32)
            voxel_fin = torch.tensor(location[4:7], dtype=torch.int32)

            name = subject_id
            sample = {
                'name': name,
                'subject_id': subject_id,
                'image': image,
                'label': label,
                'voxel_ini': voxel_ini,
                'voxel_fin': voxel_fin,
            }
            samples.append(sample)
        return samples


def niftynet_to_pytorch_format(img):
    img = np.transpose(img, (4, 0, 1, 2, 3))
    img = img[..., 0]  # remove time dimension
    return img

