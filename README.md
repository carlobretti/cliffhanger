# Find the Cliffhanger - Multi-Modal Trailerness in Soap Operas

This repo will contains the code for the MMM24 paper [Find the Cliffhanger - Multi-Modal Trailerness in Soap Operas](https://link.springer.com/chapter/10.1007/978-3-031-53308-2_15).

# Dataset preparation
The dataset can be downloaded from [here](https://drive.google.com/drive/folders/13nvqj-3TV1zy8bIA9SY9StswxKVqNOn3?usp=sharing) and the GTST folder should be placed in the same folder as this file. The dataset contains extracted features for the four possible streams: visual feats at a clip level, visual feats at a shot level, textual feats at a clip level, and textual feats at a shot level.

# Trailerness Transformer

TL;DR If you'd like to reproduce our main results:

Create the conda environment with mamba/conda with `conda env create -f environment.yml`.

Then, simply run 
```
python main.py -m datamodule.as_semantic=False,True datamodule.as_shots=False,True general.seed=10,20,30,40,50
```

Then compute late fusion predictions with `python late_fusion_frame_level.py`. This requires a WandB account that needs to be specified in `late_fusion_frame_level.py`. Finally, to get the results table, run `python tables.py`.
