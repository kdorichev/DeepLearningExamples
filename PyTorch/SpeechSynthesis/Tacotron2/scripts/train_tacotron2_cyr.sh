DATASET="VN_40h"
TEST="20210224-bs_20-ga_6-1_10sec"
BS=24
GA=6

python -m multiproc train.py -o runs/"$TEST" -d "$DATASET" --cudnn-enable --cudnn-benchmark --text-cleaners 'russian_cleaner2' --training-files  "$DATASET"/mel_dur_pitch_train_filelist.txt --validation-files "$DATASET"/mel_dur_pitch_valid_filelist.txt -m Tacotron2 --epochs 1011 --epochs-per-checkpoint 20 -lr 0.001 -bs "$BS" --gradient-accumulation-steps "$GA" --mel-fmax 12000 --log-file runs/"$TEST"/nvlog.json --load-mel-from-disk --amp --resume-from-last
