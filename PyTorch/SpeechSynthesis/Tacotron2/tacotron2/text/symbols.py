# Adapted from
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch

"""Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text
that has been run through Unidecode. For other data, you can modify _characters.
See TRAINING_DATA.md for details.

from https://github.com/keithito/tacotron
"""

_pad        = '_'
_punctuation = '!?,.:;– …'
_special = '+-'
_letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters)  # + _arpabet
pad_idx = 0
