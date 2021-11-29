python resize.py --max-size 192
python preprocess.py --use-resize
python train.py --use-resize --epochs 10 --batch-size 1 -lr 1e-5
