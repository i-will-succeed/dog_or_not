#!/usr/bin/env python
# coding: utf-8

# Usage: python3 learn.py

# This program trains the model of dogs and non-dogs

# After training, run predict.py [filename] on the model.

import os
from fastcore.all import *
from fastai.vision.all import *


# ## Is it a Dog? or something else...


## Step 2: Train our model

# We will download our dataset of animal images from the web starting with Dogs as it needs to be our biggest dataset, followed by other animals.


# Dog needs biggest data set
dog_search = 'dog',
path = Path('dog_or_not')
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"Images that failed: ${len(failed)}")

# To train a model, we'll need `DataLoaders`, which is an object that contains a *training set* (the images used to create a model) and a *validation set* (the images used to check the accuracy of a model -- not used during training). In `fastai` we can create that easily using a `DataBlock`, and view sample images from it:

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(8)

learn.save("saved_model")


# Here what each of the `DataBlock` parameters means:
# 
#     blocks=(ImageBlock, CategoryBlock),
# 
# The inputs to our model are images, and the outputs are categories (in this case, "bird" or "forest").
# 
#     get_items=get_image_files, 
# 
# To find all the inputs to our model, run the `get_image_files` function (which returns a list of all image files in a path).
# 
#     splitter=RandomSplitter(valid_pct=0.2, seed=42),
# 
# Split the data into training and validation sets randomly, using 20% of the data for the validation set.
# 
#     get_y=parent_label,
# 
# The labels (`y` values) is the name of the `parent` of each file (i.e. the name of the folder they're in, which will be *bird* or *forest*).
# 
#     item_tfms=[Resize(192, method='squish')]
# 
# Before training, resize each image to 192x192 pixels by "squishing" it (as opposed to cropping it).

# Now we're ready to train our model. The fastest widely used computer vision model is `resnet18`. You can train this in a few minutes, even on a CPU! (On a GPU, it generally takes under 10 seconds...)
# 
# `fastai` comes with a helpful `fine_tune()` method which automatically uses best practices for fine tuning a pre-trained model, so we'll use that.


