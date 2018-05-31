
## Overview
In this project, classifiers have been developed for 1) product category and 2) product classifiction in grocery store applications. With product category classification, images are classified with labels corresponding to generic food categories found in grocery stores (i.e. cereal, coffee, tea, etc.), whereas product classification labels images with specific product names (i.e. Kellog's Cereal, Folger's coffee, etc.).

The contents of this repository are as follows:
- scripts: python scripts that train and test both product / product category classifiers using Keras with TensorFlow backend
- data: 
  - product category: Freiburg Groceries [1]
  - product: zip files corresponding to product images taken in a laboratory environment 
- results: sample images and tables that show results of various architecture and hyperparameter choices
- weights: sample weights that can be used to evaluate trained models

## Product Category Classification

## Product Classification Using Siamese Networks

## References
[1]: P. Jund, N. Abdo, A. Eitel, and W. Burgard, “The Freiburg Groceries Dataset,” 2016.

[2]: G. Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile
Vision Applications,” 2017.

[3]: C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the Inception
Architecture for Computer Vision,” 2015.

[4]: C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi, “Inception-v4, Inception-ResNet and
the Impact of Residual Connections on Learning,” 2016.

[5]: K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in
2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp.
770–778.
