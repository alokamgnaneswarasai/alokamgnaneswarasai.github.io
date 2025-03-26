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

<!--more-->

## Magnitude of a Vector

### Properties of a Norm

For a vector norm $ \|\mathbf{v}\| $, the following properties hold:

1. **Non-negativity**: $ \|\mathbf{v}\| \geq 0 $, and $ \|\mathbf{v}\| = 0 $ if and only if $ \mathbf{v} = \mathbf{0} $.
2. **Absolute Scalability**: For any scalar $ \alpha $,

$$
\|\alpha \mathbf{v}\| = |\alpha| \|\mathbf{v}\|
$$

3. **Triangle Inequality**: $ \|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\| $.
4. **Norm Definiteness**: If $ \|\mathbf{v}\| = 0 $, then $ \mathbf{v} $ must be the zero vector.

### Euclidean Norm (L2 Norm)

The **magnitude (or norm)** of a vector $ \mathbf{v} $ is given by the Euclidean norm:

$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
$$

## Proof that the Euclidean Norm is a Valid Norm

### 1. Non-negativity

$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2} \geq 0
$$

As $ v_i^2 \geq 0 $ for all $ i $, it follows that $ \|\mathbf{v}\|_2 \geq 0 $. If $ \|\mathbf{v}\|_2 = 0 $, then:

$$
v_1^2 + v_2^2 + \dots + v_n^2 = 0
$$

Since each term is non-negative, $ v_i = 0 $ for all $ i $, implying $ \mathbf{v} = \mathbf{0} $.

### 2. Absolute Scalability

For any scalar $ \alpha $,

$$
\begin{align*}
\|\alpha \mathbf{v}\|_2 &= \sqrt{(\alpha v_1)^2 + (\alpha v_2)^2 + \dots + (\alpha v_n)^2} \\
&= \sqrt{\alpha^2 (v_1^2 + v_2^2 + \dots + v_n^2)} \\
&= |\alpha| \sqrt{v_1^2 + v_2^2 + \dots + v_n^2} \\
&= |\alpha| \|\mathbf{v}\|_2
\end{align*}
$$

### 3. Triangle Inequality

For vectors $ \mathbf{u}, \mathbf{v} \in \mathbb{R}^n $,

$$
\begin{align*}
\|\mathbf{u} + \mathbf{v}\|_2^2 &= \sum_{i=1}^{n} (u_i + v_i)^2 \\
&= \sum_{i=1}^{n} (u_i^2 + 2 u_i v_i + v_i^2) \\
&= \|\mathbf{u}\|_2^2 + 2 \sum_{i=1}^{n} u_i v_i + \|\mathbf{v}\|_2^2
\end{align*}
$$

By the Cauchy-Schwarz inequality:

$$
\sum_{i=1}^{n} u_i v_i \leq \|\mathbf{u}\|_2 \|\mathbf{v}\|_2
$$

Thus,

$$
\begin{align*}
\|\mathbf{u} + \mathbf{v}\|_2^2 &\leq \|\mathbf{u}\|_2^2 + 2 \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 + \|\mathbf{v}\|_2^2 \\
&= (\|\mathbf{u}\|_2 + \|\mathbf{v}\|_2)^2
\end{align*}
$$

Taking the square root:

$$
\|\mathbf{u} + \mathbf{v}\|_2 \leq \|\mathbf{u}\|_2 + \|\mathbf{v}\|_2
$$

### 4. Norm Definiteness

Follows from non-negativity: $ \|\mathbf{v}\|_2 = 0 $ if and only if $ \mathbf{v} = \mathbf{0} $.

## Homework: Verify Other Norms

Now that we’ve proved the Euclidean norm (L2 norm) is valid, let’s explore other common norms. Try proving these yourself!

### L1 Norm

The L1 norm (or Manhattan norm) of a vector $ \mathbf{v} = [v_1, v_2, \dots, v_n] $ is:

$$
\|\mathbf{v}\|_1 = |v_1| + |v_2| + \dots + |v_n|
$$

**Exercise**: Verify that $ \|\mathbf{v}\|_1 $ satisfies:

1. Non-negativity: $ \|\mathbf{v}\|_1 \geq 0 $.
2. Absolute scalability: $ \|\alpha \mathbf{v}\|_1 = |\alpha| \|\mathbf{v}\|_1 $.
3. Triangle inequality: $ \|\mathbf{u} + \mathbf{v}\|_1 \leq \|\mathbf{u}\|_1 + \|\mathbf{v}\|_1 $.
4. Definiteness: $ \|\mathbf{v}\|_1 = 0 $ if and only if $ \mathbf{v} = \mathbf{0} $.

*Hint*: Use the triangle inequality for real numbers ($ |a + b| \leq |a| + |b| $) for property 3.

### L∞ Norm

The L∞ norm (or maximum norm) of a vector $ \mathbf{v} = [v_1, v_2, \dots, v_n] $ is:

$$
\|\mathbf{v}\|_\infty = \max_{i} |v_i|
$$

**Exercise**: Verify that $ \|\mathbf{v}\|_\infty $ satisfies:

1. Non-negativity: $ \|\mathbf{v}\|_\infty \geq 0 $.
2. Absolute scalability: $ \|\alpha \mathbf{v}\|_\infty = |\alpha| \|\mathbf{v}\|_\infty $.
3. Triangle inequality: $ \|\mathbf{u} + \mathbf{v}\|_\infty \leq \|\mathbf{u}\|_\infty + \|\mathbf{v}\|_\infty $.
4. Definiteness: $ \|\mathbf{v}\|_\infty = 0 $ if and only if $ \mathbf{v} = \mathbf{0} $.

*Hint*: Consider how the maximum of a sum relates to the sum of maximums for property 3.

## Challenge: Is the Minimum Norm Valid?

Now, suppose we define a new "norm" by replacing $ \max $ with $ \min $ in the L∞ norm:

$$
\|\mathbf{v}\|_\text{min} = \min_{i} |v_i|
$$

**Question**: Does $ \|\mathbf{v}\|_\text{min} $ satisfy all four norm properties?

- Test non-negativity, absolute scalability, triangle inequality, and definiteness.
- Try a simple example, like $ \mathbf{u} = [1, 0] $ and $ \mathbf{v} = [0, 1] $, to check the triangle inequality.
- What about definiteness? Does $ \|\mathbf{v}\|_\text{min} = 0 $ imply $ \mathbf{v} = \mathbf{0} $?

---
