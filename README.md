# LearningToCountEverything

## Environment

```bash
vim ~/.config/conda/cv.yml
```

```yml
name: CV
channels:
  - pytorch
  - conda-forge
dependencies:
  # PyTorch + CUDA
  - pytorch==1.10
  - torchvision==0.11.1
  - cudatoolkit=11.3

  # Others
  # - jupyter
  - opencv
  - matplotlib
  - notebook
  - tqdm
```

```
conda update conda -y
conda update --all -y
conda env create -f ~/.config/conda/cv.yml
```

## Preprocessing

### Use resized images and densities (`generated/`)

```bash
# Proportionally zoom images/densities, so that max(H, W) <= args.max_size,
# Pad with 0s so that the shape of each image/density = max_size * max_size
# Output:
#   - generated/resized_images/
#   - generated/resized_densities/
# Record the resized images and resized bboxes coordinates
# Output:
#   - generated/image_coords.json (resized image coords)
#   - generated/bboxes_coords.json (resized bboxes coords)
python resize.py --max-size 192 --use-pad

# Preprocess generated/resized_images and generated/resized_densities
# Then output the result to
#   - generated/preprocessed_images/
#   - generated/preprocessed_densities/
python preprocess.py --use-resized
```

### Use original images and densities (`data/`)

```bash
python preprocess.py
```

## Train

```bash
python train.py --use-resized --epochs 10 --batch-size 8 -lr 1e-5
```
