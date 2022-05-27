import librosa
import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import Dataset

from utils.text_processing import encode_text
from utils.cmudict import CMUDict
from utils.utils import TacotronSTFT

class LJSpeechTextMEL(Dataset):

    def __init__(self, opts):
        self.wav_text_pairs = self.read_metadata(opts.metadata_file)
        self.cmudict = CMUDict(opts.cmudict_path)
        self.add_noise = opts.add_noise
        self.stft = TacotronSTFT(
            opts.filter_length, opts.hop_length, opts.win_length,
            opts.n_mel_channels, opts.sampling_rate, opts.mel_fmin,
            opts.mel_fmax
        )

        self.file_name = os.path.join(opts.data_root, 'wavs/{}.wav')

    def load_wav_as_tensor(self, _wav_file_):
        (sig, sampling_rate) = librosa.core.load(_wav_file_, sr=None, mono=True,  dtype=np.float32)

        return torch.FloatTensor(sig), sampling_rate

    def read_metadata(self, metadata_file):

        cols = ['id', 'transcription', 'normalized']
        df = pd.read_csv(metadata_file, sep='|', header=None, names=cols, index_col=False)
        return df.to_numpy()

    def load_mel_text(self, wav_path, text):

        # Load text
        text_norm = encode_text(text, self.cmudict)
        text_norm = torch.IntTensor(text_norm)

        # Load Audio
        audio_norm, sampling_rate = self.load_wav_as_tensor(wav_path)

        if self.add_noise:
            audio_norm = audio_norm + torch.rand_like(audio_norm)

        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return (text_norm, melspec)

    def __getitem__(self, idx):
        _sample_wav_text_pair = self.wav_text_pairs[idx]
        _wav_file_path = self.file_name.format(_sample_wav_text_pair[0])
        _text =  _sample_wav_text_pair[1]

        return self.load_mel_text(_wav_file_path, _text)

    def __len__(self):
        return self.wav_text_pairs.shape[0]


class LJCollate():

    # Pad zero
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        inp_length, ids = torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_len = inp_length[0]

        padded_text = torch.LongTensor(len(batch), max_len)
        padded_text.zero_()

        for i in range(len(ids)):
            text = batch[ids[i]][0]
            padded_text[i, :text.size(0)] = text

        # Right zero-pad
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step

        padded_mel = torch.FloatTensor(len(batch), num_mels, max_target_len)
        padded_mel.zero_()

        out_len = torch.LongTensor(len(batch))
        for i in range(len(ids)):
            mel = batch[ids[i]][1]
            padded_mel[i, :, :mel.size(1)] = mel
            out_len[i] = mel.size(1)

        return padded_text, inp_length, padded_mel, out_len