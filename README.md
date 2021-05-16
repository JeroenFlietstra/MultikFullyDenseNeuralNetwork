# Multik Fully-Dense Neural Network From Scratch

Fully connected neural network from scratch in Kotlin with Multik.

This project has been created to expirement with the [Kotlin/Multik](https://github.com/Kotlin/multik) library for linear algebra.

## main.kt

The `main()` function contains example code of how to initialize, train and test the model. This repository only contains the code for binary classification. The example 
that is being used here trains the network to classify 3 input floats on whether the sum of these floats will be a positive or a negative number.

Output from the `main()` function will display the following:

```
Total inputs with positive sum: 492
Total inputs with negative sum: 508

Epoch: 1 - Avg. Loss: 0.44435581752232145
Epoch: 2 - Avg. Loss: 0.2780377851213728
Epoch: 3 - Avg. Loss: 0.03956399372645787
Epoch: 4 - Avg. Loss: 0.029673006875174386
Epoch: 5 - Avg. Loss: 0.019782022748674664

Confusion matrix:
    0    1
0 [[139, 0],
1 [0, 161]]

Accuracy: 100.0%
```
