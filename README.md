# LearningToCountEverything

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
