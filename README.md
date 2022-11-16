# ML Cats vs Dogs
Team project for statistical methods exam

# Instructions

Neural Networks

Use Tensorflow 2 to train a neural network for the binary classification of cats and dogs based on images from this dataset. Images must be transformed from JPG to RGB (or grayscale) pixel values and scaled down. Experiment with different network architectures and training parameters documenting their influence of the final predictive performance. Use 5-fold cross validation to compute your risk estimates. While the training loss can be chosen freely, the reported cross-validated estimates must be computed according to the zero-one loss.

# Content
Dataset may contain corrupted images, see https://github.com/tensorflow/datasets/issues/2188

1. Check which images are incompatible with Tensorflow2 by running ```filter.py```; a ```files.to.delete.txt``` will be generated.
3. You can remove those images by using the ```utilities``` module as follow:

```python
    import utilities as ff
    ff.delete_from_list()
```

# Notes

## https://www.quora.com/Does-TensorFlow-have-an-implementation-of-Cross-Validation-one-can-use
Cross validation is most useful when the dataset is relatively small (hundreds, maybe thousands of training examples).
When the dataset is large, training can already take hours.
Using cross validation will result in multiplying the training time by the number of folds, but won’t give you much benefit.

The reason cross validation is useful is because when you have a small number of examples to work with, you run the risk of “missing something important” when you’re unable to train on 20% of your data because you’re using it for testing.
But as the dataset size increases, you are statistically less likely to encounter this problem.

Deep learning models only really make sense when you have a lot of training data to work with.
So if your dataset is small enough that you can afford to use cross validation, you’re probably overfitting and should use a less complex model.

## https://towardsdatascience.com/train-test-split-c3eed34f763b

We must not yet assume that performance on the validation data is the “real” performance of each model.
At this point we introduce the last step, accompanied by the test dataset.
We use the model to classify the test data and calculate the test error metrics.

This sounds like we’re doing the same thing all over again, just using the test set — but why?
The best model (in terms of highest accuracy) after validation may be good, but there it is also likely that one model may have benefited from being more suitable for the random pattern in the validation data — the random effects — and this, only by chance.

This is not what we want to measure our models on, we do not want to choose a model, assuming it is better, but in reality the model only outperformed simply due to luck. For this reason, the test set introduces another layer that minimizes chances that a model benefits from pure randomness.