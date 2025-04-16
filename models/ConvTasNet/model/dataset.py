import os
import torchaudio
from torch.utils.data import Dataset
import torch

class Dataset(Dataset):
    def __init__(self, mixtures_dir, sources_dir=None, sample_rate=16000, segment_length=4.0, transform=None, is_test=False):
        """
        Args:
            mixtures_dir (str): Directory with mixture audio files.
            sources_dir (str, optional): Directory with source audio files. Set to None for test mode.
            sample_rate (int, optional): Sample rate of audio files.
            segment_length (float, optional): Length of the audio segment in seconds. Ignored in test mode.
            transform (callable, optional): Optional transform to be applied on an audio sample.
            is_test (bool, optional): If True, operates in test mode and loads the entire file.
        """
        self.mixtures_dir = mixtures_dir
        self.sources_dir = sources_dir
        self.sample_rate = sample_rate
        self.segment_length = int(sample_rate * segment_length)
        self.transform = transform
        self.is_test = is_test

        # List and sort mixture files
        self.mixtures = sorted([os.path.join(mixtures_dir, f) for f in os.listdir(mixtures_dir)
                                if f.endswith('.wav')])

        if not self.is_test:
            # List and sort source files (only if in training/validation mode)
            self.sources = sorted([os.path.join(sources_dir, f) for f in os.listdir(sources_dir)
                                   if f.endswith('.wav')])
            assert len(self.mixtures) == len(self.sources), "Number of mixtures and sources must be equal."
        
        if not self.is_test:
            # Segment the files only in training/validation mode
            self.segments = []
            for mix_path in self.mixtures:
                info = torchaudio.info(mix_path)
                num_samples = info.num_frames
                num_segments = (num_samples + self.segment_length - 1) // self.segment_length
                src_path = self.sources[self.mixtures.index(mix_path)]
                for i in range(num_segments):
                    start_idx = i * self.segment_length
                    self.segments.append((mix_path, src_path, start_idx))
        else:
            # In test mode, just store the mixture paths (no segmentation)
            self.segments = [(mix_path, None, None) for mix_path in self.mixtures]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        mix_path, src_path, start_idx = self.segments[idx]
        
        if self.is_test:
            # In test mode, load the entire audio file
            mix_waveform, _ = torchaudio.load(mix_path)
            if self.transform:
                mix_waveform = self.transform(mix_waveform)
            return mix_waveform, mix_path
        else:
            # In training/validation mode, load segments
            mix_waveform, _ = self.load_segment(mix_path, start_idx)
            src_waveform, _ = self.load_segment(src_path, start_idx)

            if self.transform:
                mix_waveform = self.transform(mix_waveform)
                src_waveform = self.transform(src_waveform)

            # Ensure both waveforms have the same length by padding (for consistency)
            mix_waveform, src_waveform = self.pad_waveforms(mix_waveform, src_waveform)

            return mix_waveform, src_waveform

    def load_segment(self, file_path, start_idx):
        waveform, sample_rate = torchaudio.load(file_path)
        end_idx = start_idx + self.segment_length
        waveform_segment = waveform[:, start_idx:end_idx]
        return waveform_segment, sample_rate

    def pad_waveforms(self, mix_waveform, src_waveform):
        max_len = max(mix_waveform.size(1), src_waveform.size(1))
        mix_waveform = torch.nn.functional.pad(mix_waveform, (0, max_len - mix_waveform.size(1)))
        src_waveform = torch.nn.functional.pad(src_waveform, (0, max_len - src_waveform.size(1)))
        return mix_waveform, src_waveform


def collate_fn(batch):
    if isinstance(batch[0][1], str):  # Test mode, second element is filename
        mix_waveforms = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
    else:
        mix_waveforms = [item[0] for item in batch]
        src_waveforms = [item[1] for item in batch]

    max_len = max([waveform.size(1) for waveform in mix_waveforms])

    mix_waveforms_padded = [torch.nn.functional.pad(w, (0, max_len - w.size(1))) for w in mix_waveforms]

    mix_waveforms_batch = torch.stack(mix_waveforms_padded)

    if isinstance(batch[0][1], str):  # Test mode
        return mix_waveforms_batch, filenames
    else:  # Training/Validation mode
        src_waveforms_padded = [torch.nn.functional.pad(w, (0, max_len - w.size(1))) for w in src_waveforms]
        src_waveforms_batch = torch.stack(src_waveforms_padded)
        return mix_waveforms_batch, src_waveforms_batch
