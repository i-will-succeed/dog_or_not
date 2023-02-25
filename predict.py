
# Usage: python3 predict.py [filepath]
# Loads the model saved in models/saved_model.pth
# And runs the prediction on the model and the filepath

import os
import sys
from fastcore.all import *
from fastai.vision.all import *
import warnings

warnings.filterwarnings("ignore")

path = Path('dog_or_not')

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=24)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.load("saved_model")

arg1 = sys.argv[1]
print(f"Predicting image ${arg1}")
image_path = Path(arg1)
is_dog,_,probs = learn.predict(image_path)
print(f"This looks like a: {is_dog}.")
print(f"Probability it's a dog: {probs[2]:.4f}")

