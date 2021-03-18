TEST="test2"
BS=128
GA=1

python -m multiproc train.py -o runs/"$TEST" -d minidataset --cudnn-enable --cudnn-benchmark --text-cleaners 'russian_cleaner2' --training-files  minidataset/mel_dur_pitch_train_filelist.txt --validation-files minidataset/mel_dur_pitch_valid_filelist.txt -m Tacotron2 --epochs 1 --epochs-per-checkpoint 1 -lr 0.001 -bs "$BS" --gradient-accumulation-steps "$GA" --mel-fmax 12000 --log-file runs/"$TEST"/nvlog.json --load-mel-from-disk --amp
