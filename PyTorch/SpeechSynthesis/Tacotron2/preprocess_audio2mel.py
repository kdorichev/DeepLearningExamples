import argparse
from argparse import ArgumentParser
import torch

from tacotron2.data_function import TextMelLoader
from common.utils import load_filepaths_and_text

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-files', required=True,
                        type=str, help='Path to filelist with audio paths and text')
    parser.add_argument('--mel-files', required=True,
                        type=str, help='Path to filelist with mel paths and text')
    parser.add_argument('--text-cleaners', nargs='*',
                        default=['russian_cleaner2'], type=str,
                        help='Type of text cleaners for input text. Default: russian_cleaner2')
    parser.add_argument('--max-wav-value', default=1.0, type=float,
                        help='Maximum audiowave value to divide on for normalization')
    parser.add_argument('--sampling-rate', default=22050, type=int,
                        help='Sampling rate. Default: 22050')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length. Default: 1024')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length. Default: 256')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length. Default: 1024')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency. Default: 0.0')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency. Default: 8000.0')
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms. Default: 80')
    parser.add_argument('--load-mel-from-disk', action='store_true', default=False, 
                        help='To load pre-calculated mels from disk. Default: False')

    return parser


def audio2mel(dataset_path: str, audiopaths_and_text: str, melpaths_and_text: str,
              args: ArgumentParser)-> None:
    """Create mel spectrograms on disk from audio files.

    Args:
        dataset_path (str): Path to dataset
        audiopaths_and_text (str): Path to filelist with audio paths and text
        melpaths_and_text (str): Path to filelist with mel paths and text
        args (ArgumentParser): Namespace with arguments
    """

    melpaths_and_text_list = load_filepaths_and_text(dataset_path, melpaths_and_text)
    audiopaths_and_text_list = load_filepaths_and_text(dataset_path, audiopaths_and_text)
    data_loader = TextMelLoader(dataset_path, audiopaths_and_text, args)

    for i, melpath_and_text in enumerate(melpaths_and_text_list):
        if i%100 == 0:
            print("done", i, "/", len(melpaths_and_text_list))
        mel = data_loader.get_mel(audiopaths_and_text_list[i][0])
        torch.save(mel, melpath_and_text[0])

def main():

    parser = argparse.ArgumentParser(description='PyTorch Tacotron2 Training')
    parser = parse_args(parser)
    args = parser.parse_args()

    audio2mel(args.dataset_path, args.wav_files, args.mel_files, args)

if __name__ == '__main__':
    main()
