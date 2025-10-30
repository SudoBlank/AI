import numpy as np
import math
from typing import List, Dict, Any
import pickle
import os

class Tensor:
    """Proper tensor implementation with autograd"""
    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other))
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other))
        
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        # Topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Backward pass
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

class LinearLayer:
    """Proper linear layer with Xavier initialization"""
    def __init__(self, input_size, output_size):
        scale = math.sqrt(2.0 / (input_size + output_size))
        self.weights = Tensor(np.random.randn(input_size, output_size) * scale, requires_grad=True)
        self.bias = Tensor(np.zeros(output_size), requires_grad=True)
    
    def __call__(self, x):
        return x @ self.weights + self.bias
    
    def parameters(self):
        return [self.weights, self.bias]

class NeuralCore:
    """Core neural network that can actually learn"""
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(LinearLayer(layer_sizes[i], layer_sizes[i + 1]))
    
    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on output
                x = self.relu(x)
        return x
    
    def relu(self, x):
        data = np.maximum(0, x.data)
        requires_grad = x.requires_grad
        out = Tensor(data, requires_grad, _children=(x,))
        
        def _backward():
            if x.requires_grad:
                x.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)