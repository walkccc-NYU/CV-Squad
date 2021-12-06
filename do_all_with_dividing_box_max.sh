python resize.py --max-size 192 --use-pad
python preprocess.py --use-resized -n box_max
python train.py --use-resize --epochs 1000 --batch-size 8 -lr 1e-5 --log-interval 1 -n box_max
