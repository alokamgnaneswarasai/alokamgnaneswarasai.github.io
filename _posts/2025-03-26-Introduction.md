---
layout: post
title: Introduction to vectors in Linear Algebra
---


## Definition of a Vector

In linear algebra, a **vector** is an element of a vector space. Mathematically, an **n-dimensional vector** is represented as:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n
$$

where each $v_i $ is a real number, and${R}^n$ denotes an n-dimensional real-valued vector space.

## Magnitude of a Vector

The **magnitude (or norm)** of a vector $$\mathbf{v}$$ is given by the Euclidean norm:

$$
||\mathbf{v}|| = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
$$

This represents the length of the vector in an n-dimensional space.

## Vector Operations

Given two vectors $ \mathbf{u} $ and $ \mathbf{v} $, the following operations can be performed:

### 1. Vector Addition

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$

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

where $ c_1, c_2, \dots, c_k $ are scalar coefficients. A set of vectors is said to be **linearly dependent** if at least one of them can be written as a combination of the others; otherwise, they are **linearly independent**.

---
