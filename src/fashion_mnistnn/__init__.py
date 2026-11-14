"""Provide a neural network for Fashion-MNIST clothing classification.

Fashion-MNIST uses the same 28x28 IDX layout as classic MNIST, with 10
fashion classes instead of handwritten digits.

Architecture:
- Input: 784 (28x28 flattened)
- Hidden Layer 1: 512 units with ReLU and dropout 0.1
- Hidden Layer 2: 256 units with ReLU and dropout 0.2
- Hidden Layer 3: 128 units with ReLU and dropout 0.2
- Output Layer: 10 units with Softmax
"""
