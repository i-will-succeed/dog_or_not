#!/usr/bin/env python
# coding: utf-8

# Usage python3 fetch_images.py 

# downloads training images for dog recognition


# ## Is it a Dog? or something else...
# `!pip install -Uqq <libraries>` upgrades to the latest version of <libraries>
import os
# get_ipython().system('pip install -Uqq fastai duckduckgo_search')



# ![image.png](attachment:a0483178-c30e-4fdd-b2c2-349e130ab260.png)

# But today, we can do exactly that, in just a few minutes, using entirely free resources!
# 
# The basic steps we'll take are:
# 
# 1. Use DuckDuckGo to search for images of "dog photos"
# 1. Use DuckDuckGo to search for images of "non-dog common house pets photos"
# 1. Fine-tune a pretrained neural network to recognise these two groups
# 1. Try running this model on a picture of a dog and see if it works.
# ## Step 1: Download images of dogs and non-dogs

# In[2]:


from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')


#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).
urls = search_images('dog photos', max_images=1)

from fastdownload import download_url
download_url(urls[0], dest, show_progress=False)
from fastai.vision.all import *


# Our searches seem to be giving reasonable results, so let's grab a few examples of each of 'dog','cat','skunk','rabbit', 'guinea pig', and 'ferret' photos, and save each group of photos to a different folder (I'm also trying to grab a range of lighting conditions here):

# ## Step 2: Train our model

# We will download our dataset of animal images from the web starting with Dogs as it needs to be our biggest dataset, followed by other animals.

# Dog needs biggest data set
dog_search = 'dog',
path = Path('dog_or_not')
from time import sleep

for o in dog_search:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(5)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'small {o} photo'))
    sleep(5)
    download_images(dest, urls=search_images(f'big {o} photo'))
    sleep(5)
    resize_images(path/o, max_size=400, dest=path/o)
    
# Compare to other animals for testing
searches2 = 'cat','skunk','rabbit', 'guinea pig', 'ferret', 'coyote', 'honey badger', 'wolverine animal', 'wolf', 'raccoon'

for i in searches2:
    dest = (path/i)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{i} photo'))
    sleep(5) # Pause between searches to avoid over-loading server
    resize_images(path/i, max_size=400, dest=path/i)


# Some photos might not download correctly which could cause our model training to fail, so we'll remove them:

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"failed images: ${len(failed)}")
