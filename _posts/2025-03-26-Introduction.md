---
layout: post
title: Introduction to vectors in Linear Algebra
---


{%- include mathjax.html -%}

## Definition of a Vector

In linear algebra, a **vector** is an element of a vector space. Mathematically, an **n-dimensional vector** is represented as:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n
$$

where each $ v_i $ is a real number, and $ \mathbb{R}^n $ denotes an n-dimensional real-valued vector space.


## Vector Operations

Given two vectors $ \mathbf{u} $ and $ \mathbf{v} $, the following operations can be performed:

### 1. Vector Addition

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$


Similarly, we can perform vector subtraction $ \mathbf{u} - \mathbf{v} $ by rewriting it as $ (\mathbf{u}) + (\mathbf{-v}) $.


### 2. Scalar Multiplication

For a scalar $ \alpha $:

$$
\alpha \mathbf{v} = \begin{bmatrix} \alpha v_1 \\ \alpha v_2 \\ \vdots \\ \alpha v_n \end{bmatrix}
$$

### 3. Linear Combination of Vectors

A **linear combination** of vectors $ \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k $ is expressed as:

$$
\mathbf{w} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k
$$

where $ c_1, c_2, \dots, c_k $ are scalar coefficients. 


## Magnitude of a Vector

### Properties of a Norm

For a vector norm $ \|\mathbf{v}\| $, the following properties hold:

1. **Non-negativity**: $ \|\mathbf{v}\| \geq 0 $, and $ \|\mathbf{v}\| = 0 $ if and only if $ \mathbf{v} = \mathbf{0} $.
2. **Absolute Scalability**: For any scalar $ \alpha $, $ \|\alpha \mathbf{v}\| = |\alpha| \|\mathbf{v}\| $.
3. **Triangle Inequality**: $ \|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\| $.
4. **Norm Definiteness**: If $ \|\mathbf{v}\| = 0 $, then $ \mathbf{v} $ must be the zero vector.

### Euclidean Norm (L2 Norm)

The **magnitude (or norm)** of a vector $ \mathbf{v} $ is given by the Euclidean norm:

$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
$$


## Proof that the Euclidean Norm is a Valid Norm

### 1. Non-negativity:
$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2} \geq 0
$$
as $ v_i^2 \geq 0 $ for all $ i $, it follows that $ \|\mathbf{v}\|_2 \geq 0 $.  
If $ \|\mathbf{v}\|_2 = 0 $, then:
$$
v_1^2 + v_2^2 + \dots + v_n^2 = 0
$$
Since each term is non-negative, $ v_i = 0 $ for all $ i $, implying $ \mathbf{v} = \mathbf{0} $.

### 2. Absolute Scalability:
For any scalar $ \alpha $,
$$
\|\alpha \mathbf{v}\|_2 = \sqrt{(\alpha v_1)^2 + (\alpha v_2)^2 + \dots + (\alpha v_n)^2}
$$
$$
= \sqrt{\alpha^2 (v_1^2 + v_2^2 + \dots + v_n^2)}
$$
$$
= |\alpha| \sqrt{v_1^2 + v_2^2 + \dots + v_n^2} = |\alpha| \|\mathbf{v}\|_2.
$$

### 3. Triangle Inequality:
For vectors $ \mathbf{u}, \mathbf{v} \in \mathbb{R}^n $,
$$
\|\mathbf{u} + \mathbf{v}\|_2^2 = \sum_{i=1}^{n} (u_i + v_i)^2
$$
$$
= \sum_{i=1}^{n} (u_i^2 + 2 u_i v_i + v_i^2)
$$
$$
= \|\mathbf{u}\|_2^2 + 2 \sum_{i=1}^{n} u_i v_i + \|\mathbf{v}\|_2^2.
$$
By the Cauchy-Schwarz inequality:
$$
\sum_{i=1}^{n} u_i v_i \leq \|\mathbf{u}\|_2 \|\mathbf{v}\|_2.
$$
Thus,
$$
\|\mathbf{u} + \mathbf{v}\|_2^2 \leq (\|\mathbf{u}\|_2 + \|\mathbf{v}\|_2)^2.
$$
Taking the square root:
$$
\|\mathbf{u} + \mathbf{v}\|_2 \leq \|\mathbf{u}\|_2 + \|\mathbf{v}\|_2.
$$

### 4. Norm Definiteness:

 Follows from non-negativity: $ \|\mathbf{v}\|_2 = 0 $ if and only if $ \mathbf{v} = \mathbf{0} $.
---
