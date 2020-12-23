#!/usr/bin/env bash

set -e

DATADIR="Voituk_Narrative_10s_no_abbrev_strange"
FILELISTSDIR="Voituk_Narrative_10s_no_abbrev_strange"

TRAINLIST="$FILELISTSDIR/train_filelist.txt"
VALLIST="$FILELISTSDIR/valid_filelist.txt"

TRAINLIST_MEL="$FILELISTSDIR/mel_dur_pitch_train_filelist.txt"
VALLIST_MEL="$FILELISTSDIR/mel_dur_pitch_valid_filelist.txt"

mkdir -p "$DATADIR/mels"
python preprocess_audio2mel.py -d "$DATADIR" --wav-files "$TRAINLIST" --mel-files "$TRAINLIST_MEL"
python preprocess_audio2mel.py -d "$DATADIR" --wav-files "$VALLIST" --mel-files "$VALLIST_MEL"	
#fi	
