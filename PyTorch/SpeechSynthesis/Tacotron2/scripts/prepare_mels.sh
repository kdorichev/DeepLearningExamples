#!/usr/bin/env bash

set -e

DATADIR="Voituk_Narrative"
FILELISTSDIR="filelists"

#TESTLIST="$FILELISTSDIR/ljs_audio_text_test_filelist.txt"
TRAINLIST="$FILELISTSDIR/Voituk_Narrative_train_filelist.txt"
VALLIST="$FILELISTSDIR/Voituk_Narrative_valid_filelist.txt"

#TESTLIST_MEL="$FILELISTSDIR/ljs_mel_text_test_filelist.txt"
TRAINLIST_MEL="$FILELISTSDIR/Voituk_Narrative_mel_dur_pitch_train_filelist.txt"
VALLIST_MEL="$FILELISTSDIR/Voituk_Narrative_mel_dur_pitch_valid_filelist.txt"

mkdir -p "$DATADIR/mels"
#if [ $(ls $DATADIR/mels | wc -l) -ne 13100 ]; then
python preprocess_audio2mel.py -d Voituk_Narrative --wav-files "$TRAINLIST" --mel-files "$TRAINLIST_MEL"
#    python preprocess_audio2mel.py --wav-files "$TESTLIST" --mel-files "$TESTLIST_MEL"
python preprocess_audio2mel.py -d Voituk_Narrative --wav-files "$VALLIST" --mel-files "$VALLIST_MEL"	
#fi	
