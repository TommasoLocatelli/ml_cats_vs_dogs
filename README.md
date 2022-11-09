# ML Cats vs Dogs
Team project for statistical methods exam

# Instructions

Neural Networks

Use Tensorflow 2 to train a neural network for the binary classification of cats and dogs based on images from this dataset. Images must be transformed from JPG to RGB (or grayscale) pixel values and scaled down. Experiment with different network architectures and training parameters documenting their influence of the final predictive performance. Use 5-fold cross validation to compute your risk estimates. While the training loss can be chosen freely, the reported cross-validated estimates must be computed according to the zero-one loss.

# Content
Dataset may contain corrupted images

1. Check which images are incompatible with Tensorflow2 by running ```filter.py```; a ```files.to.delete.txt``` will be generated.
3. You can remove those images by using the ```utilities``` module as follow:

```python
    import utilities as ff
    ff.delete_from_list()
```