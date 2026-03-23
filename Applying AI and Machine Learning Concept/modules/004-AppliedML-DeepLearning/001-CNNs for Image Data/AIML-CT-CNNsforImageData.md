# CNNs for Image Data

**Estimated Time:** 10 Minutes

---

## Introduction

If you have ever wondered how your phone can recognize faces or how a self-driving car tells a stop sign from a speed-limit sign, the answer almost always involves a **Convolutional Neural Network (CNN)**. CNNs are the workhorse architecture for image-related tasks, and understanding them is essential for any applied ML practitioner.

What makes CNNs special is not just that they work well on images -- it is *why* they work well. Instead of treating an image as a flat list of pixel values (the way a fully connected network would), a CNN preserves the spatial structure of the image and learns local patterns like edges, textures, and shapes. This reading will walk you through the building blocks of a CNN and explain why data augmentation is a critical companion technique.

---

## Core Concepts

### Why Fully Connected Networks Struggle with Images

Consider a modest 128x128 color image. Flattened, that is 128 * 128 * 3 = 49,152 input features. A single fully connected hidden layer of 512 neurons would require over 25 million parameters -- just for the first layer. This is wasteful because most of those connections are learning nothing useful; a pixel in the top-left corner rarely has a direct relationship with a pixel in the bottom-right corner.

CNNs solve this with two key ideas: **local connectivity** (each neuron only looks at a small patch of the input) and **weight sharing** (the same small filter is slid across the entire image).

### Convolutional Layers

A convolutional layer applies a set of small learnable filters (also called kernels) to the input. Each filter slides across the image, computing a dot product at every position, producing a **feature map**. Early layers tend to learn low-level features like edges and color gradients, while deeper layers combine those into higher-level concepts like eyes, wheels, or letters.

Key parameters you control:

- **Number of filters** -- how many different features this layer can detect (e.g., 16, 32, 64).
- **Kernel size** -- the height and width of each filter (commonly 3x3 or 5x5).
- **Stride** -- how many pixels the filter moves at each step. A stride of 1 moves one pixel at a time; a stride of 2 skips every other position, halving the output dimensions.
- **Padding** -- whether to add border pixels so the output has the same spatial dimensions as the input.

For example, a convolutional layer with 16 filters of size 3x3, stride 1, and "same" padding applied to a 32x32 RGB input produces 16 feature maps, each still 32x32. The total parameters for this layer are \(3 \times 3 \times 3 \times 16 + 16 = 448\) (kernel weights plus biases) -- dramatically fewer than a fully connected equivalent.

### Pooling Layers

After a convolutional layer, a **pooling layer** reduces the spatial dimensions. The most common variant is **max pooling**, which takes the maximum value in each small window (typically 2x2 with stride 2, halving each spatial dimension). Pooling does two things: it shrinks the feature maps (lowering computation in later layers) and it adds a degree of **translation invariance** -- the network cares less about *exactly* where a feature appears and more about *whether* it appears.

**Average pooling** takes the mean instead of the maximum and is sometimes used in later layers, especially before the final classifier.

### Putting It Together: A Simple CNN

A typical CNN stacks several rounds of **Convolution + Activation + Pooling**, then flattens the result and feeds it into one or more fully connected layers for classification:

```
Input image (3 x 32 x 32)
  -> Conv(3 -> 16, kernel=3x3, pad=1) -> ReLU -> MaxPool(2x2)
     Output: 16 x 16 x 16
  -> Conv(16 -> 32, kernel=3x3, pad=1) -> ReLU -> MaxPool(2x2)
     Output: 32 x 8 x 8
  -> Flatten
     Output: 2048
  -> FullyConnected(2048 -> 128) -> ReLU
  -> FullyConnected(128 -> num_classes)
```

This pattern -- conv-relu-pool stacks feeding into fully connected heads -- reappears in every major image model, from AlexNet to ResNet.

### Data Augmentation: Getting More from Less

Deep networks are hungry for data, and collecting labeled images is expensive. **Data augmentation** artificially expands your training set by applying random transformations -- flips, rotations, crops, color jitter -- to each image every time it is loaded. The model never sees the exact same image twice, which reduces overfitting and improves generalization.

Common augmentation techniques:

- **Random horizontal flip** -- mirrors the image left-to-right.
- **Random rotation** -- rotates by a small angle (e.g., up to 15 degrees).
- **Color jitter** -- randomly adjusts brightness, contrast, and saturation.
- **Random cropping/resizing** -- crops a random sub-region and resizes it back to the original dimensions.
- **Normalization** -- scaling pixel values to zero mean and unit variance using dataset statistics (or standard ImageNet values as a common default).

A few practical rules of thumb:

- **Always normalize** your images (use the mean and standard deviation of your dataset, or widely-used defaults).
- **Only augment the training set.** Validation and test sets should use deterministic preprocessing so your metrics are reproducible.
- **Choose augmentations that make semantic sense.** Horizontal flips are fine for natural photos but would be harmful for digit recognition (a flipped "6" looks like a "9").

### When to Use CNNs

CNNs are the right choice whenever your data has **spatial structure** -- images, spectrograms, heatmaps, even some types of tabular data arranged in grids. They are not the right tool for sequential data like text or time series (that is where RNNs and Transformers come in, which you will explore next).

---

## Connecting to Practice

In the hands-on exercise that follows, you will build and train a CNN on a real image dataset. You will experiment with different numbers of convolutional layers, kernel sizes, and augmentation strategies, and you will observe directly how these choices affect accuracy and overfitting. The architecture patterns you learn here reappear in every major image model, so this foundation will serve you well as architectures grow more sophisticated.

---

## Further Learning & Resources

### Documentation

1. [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/) -- In-depth course notes covering CNN architectures, backpropagation through convolutions, and practical tips.
2. [Deep Learning Book, Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html) -- Goodfellow, Bengio, and Courville's thorough mathematical treatment of CNNs.
3. [Wikipedia: Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) -- Comprehensive overview of CNN history, architecture variants, and applications.

### Interactive Resources

1. [CNN Explainer](https://poloclub.github.io/cnn-explainer/) -- Interactive visualization that lets you watch data flow through each layer of a CNN in real time.
2. [TensorFlow Playground](https://playground.tensorflow.org/) -- Experiment with network architectures and observe how adding layers and changing activations affects learned decision boundaries.
3. [Google Machine Learning Crash Course: Image Classification](https://developers.google.com/machine-learning/practica/image-classification) -- Guided walkthrough of building and training a CNN with embedded exercises.
