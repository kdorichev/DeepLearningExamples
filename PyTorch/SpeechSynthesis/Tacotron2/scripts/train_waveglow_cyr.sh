TEST="wg-0323"
DS="VN_40h"
BS=1
GA=4

python -m multiproc train.py -m WaveGlow -o ./runs/"$TEST" -lr 1e-4 --epochs 10 --epochs-per-checkpoint 10 --gradient-accumulation "$GA" --load-mel-from-disk -bs "$BS" --sampling-rate 24000 --segment-length 6000 --weight-decay 0 --grad-clip-thresh 3.4028234663852886e+38 --cudnn-enabled --cudnn-benchmark --log-file ./runs/"$TEST"/nvlog.json -d "$DS" --training-files "$DS"/train_filelist.txt --validation-files "$DS"/valid_filelist.txt --resume-from-last --mel-fmax 12000 
#python -m multiproc train.py -m WaveGlow -o ./output/ -lr 1e-4 --epochs 1501 -bs 4 --segment-length  8000 --weight-decay 0 --grad-clip-thresh 3.4028234663852886e+38 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json
