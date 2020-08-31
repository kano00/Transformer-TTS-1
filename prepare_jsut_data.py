import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from utils import get_spectrograms
import hyperparams as hp
import librosa


class PrepareJSUTDataset(Dataset):
    """JSUT dataset."""

    def __init__(self, txt_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        with open(txt_file) as f:
            txt = f.read().splitlines()
            self.landmarks_frame = [i.split(':') for i in txt]
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sr)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame[idx][0]) + '.wav'
        mel, mag = get_spectrograms(wav_name)

        np.save(wav_name[:-4] + '.pt', mel)
        np.save(wav_name[:-4] + '.mag', mag)

        sample = {'mel': mel, 'mag': mag}

        return sample


if __name__ == '__main__':
    dataset = PrepareJSUTDataset(os.path.join(hp.data_path, 'phoneme_id_dict.txt'), os.path.join(hp.data_path, 'wavs'))
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm

    pbar = tqdm(dataloader)
    for d in pbar:
        pass
