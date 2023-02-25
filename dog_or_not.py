#!/usr/bin/env python
# coding: utf-8

# ## Is it a Dog? or something else...

# In[1]:


# It's a good idea to ensure you're running the latest version of any libraries you need.
# `!pip install -Uqq <libraries>` upgrades to the latest version of <libraries>
# NB: You can safely ignore any warnings or errors pip spits out about running as root or incompatibilities
import os
get_ipython().system('pip install -Uqq fastai duckduckgo_search')


# In 2015 the idea of creating a computer system that could recognise birds was considered so outrageously challenging that it was the basis of [this XKCD joke](https://xkcd.com/1425/):

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


# Let's start by searching for a dachshund photo and seeing what kind of result we get. We'll start by getting URLs from a search:

# In[3]:


#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).
urls = search_images('dog photos', max_images=1)
urls[0]


# ...and then download a URL and take a look at it:

# In[4]:


from fastdownload import download_url
dest = 'dog.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)


# Now let's do the same with "cat photos":

# In[5]:


download_url(search_images('cat photos', max_images=1)[0], 'cat.jpg', show_progress=False)
Image.open('cat.jpg').to_thumb(256,256)


# Our searches seem to be giving reasonable results, so let's grab a few examples of each of 'dog','cat','skunk','rabbit', 'guinea pig', and 'ferret' photos, and save each group of photos to a different folder (I'm also trying to grab a range of lighting conditions here):

# ## Step 2: Train our model

# We will download our dataset of animal images from the web starting with Dogs as it needs to be our biggest dataset, followed by other animals.

# In[6]:


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

# In[7]:


failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)


# To train a model, we'll need `DataLoaders`, which is an object that contains a *training set* (the images used to create a model) and a *validation set* (the images used to check the accuracy of a model -- not used during training). In `fastai` we can create that easily using a `DataBlock`, and view sample images from it:

# In[8]:


dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=24)


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

# In[9]:


learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(8)


# Generally when I run this I see 100% accuracy on the validation set (although it might vary a bit from run to run).
# 
# "Fine-tuning" a model means that we're starting with a model someone else has trained using some other dataset (called the *pretrained model*), and adjusting the weights a little bit so that the model learns to recognise your particular dataset. In this case, the pretrained model was trained to recognise photos in *imagenet*, and widely-used computer vision dataset with images covering 1000 categories) For details on fine-tuning and why it's important, check out the [free fast.ai course](https://course.fast.ai/).

# ## Step 3: Use our model (and build your own!)

# Let's see what our model thinks about that dog we downloaded at the start along with any other animals in our dataset:

# In[11]:


test_imgs = Path('test_imgs')
is_dog,_,probs = learn.predict(test_imgs/'dog.jpg')
print(f"This looks like a: {is_dog}.")
print(f"Probability it's a dog: {probs[2]:.4f}")


# Feel free to explore with uploading different animals in our test_imgs folder. As of now, our model will only be able to identify the current animals: cat, dog, coyote, ferret, guinea pig, honey badger, rabbit, raccoon, skunk, wolf, and wolverine.

# Good job, resnet18. :)
# 
# So, as you see, in the space of a few years, creating computer vision classification models has gone from "so hard it's a joke" to "trivially easy and free"!
# 
# It's not just in computer vision. Thanks to deep learning, computers can now do many things which seemed impossible just a few years ago, including [creating amazing artworks](https://openai.com/dall-e-2/), and [explaining jokes](https://www.datanami.com/2022/04/22/googles-massive-new-language-model-can-explain-jokes/). It's moving so fast that even experts in the field have trouble predicting how it's going to impact society in the coming years.
# 
# One thing is clear -- it's important that we all do our best to understand this technology, because otherwise we'll get left behind!

# Now it's your turn. Click "Copy & Edit" and try creating your own image classifier using your own image searches!
# 
# If you enjoyed this, please consider clicking the "upvote" button in the top-right -- it's very encouraging to us notebook authors to know when people appreciate our work.
