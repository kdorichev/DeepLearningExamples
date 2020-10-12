# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
from typing import Optional, Tuple

import numpy as np
import torch
from librosa.core import load
from librosa.effects import trim

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


def load_wav_to_torch(full_path: str, sr: Optional[int] = 22050) -> Tuple[torch.Tensor, int]:
    """Load audio file from `full_path` with optional resamplling to `sr`.
    Args:
        full_path (str): path to audio file.
        sr (int, optional): sample rate to resample to.
    Returns:
        (torch.Tensor, sampling_rate)
    """

    data, sampling_rate = load(full_path, sr)
    return torch.from_numpy(trim(data)), sampling_rate


def load_filepaths_and_text(dataset_path: str, filename: str, split="|") -> list:
    """Return a list of tuples: (path to mel file, text).
    `filename` has either 2 or 4 fields, like:

    LJSpeech-1.1/mels/LJ003-0182.pt|The tried and the untried, ...
    mels/EHD_120770D_022.pt|durations/EHD_120770D_022.pt|pitch_char/EHD_120770D_022.pt|домысел

    Args:
        dataset_path (str): Path to root of the dataset.
        filename (str): Name of file to parse.
        split (str, optional): A symbol to split on. Defaults to "|".

    Raises:
        Exception: [description]

    Returns:
        list of tuples: (path to mel file, text)
    """
    def split_line(root, line):
        parts = line.strip().split(split)
        if (len(parts) != 2) and (len(parts) != 4):
            raise Exception(f"incorrect line format for file: {filename}")
        text_idx = 1 if len(parts) == 2 else 3
        path = os.path.join(root, parts[0])
        text = parts[text_idx]
        return path, text

    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [split_line(dataset_path, line) for line in f]

    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x
