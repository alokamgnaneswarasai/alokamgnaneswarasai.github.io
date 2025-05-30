---
layout: post
title: Introduction to vectors in Linear Algebra
tags: [linear algebra, vectors, mathematics]
author: Gnaneswara Sai
---
<!-- {%- include mathjax.html -%} -->


## Definition of a Vector

In linear algebra, a **vector** is an element of a **vector space**. A vector space $$ V $$ is a set equipped with two operations—vector addition and scalar multiplication—that satisfy certain axioms (e.g., associativity, commutativity, distributivity). These operations are defined over a **field** $$ F $$, such as the real numbers $$ \mathbb{R} $$, complex numbers $$ \mathbb{C} $$, or any other field (e.g., rational numbers $$ \mathbb{Q} $$ or finite fields $$ \mathbb{F}_p $$). Thus, a vector $$ \mathbf{v} $$ is any element $$ \mathbf{v} \in V $$, where $$ V $$ is a vector space over $$ F $$.

For practical purposes, vectors are often represented with coordinates once a basis is chosen. Below are common representations in the context of the vector space $$ \mathbb{R}^n $$ over the field $$ \mathbb{R} $$, though vectors can take other forms depending on the space (e.g., polynomials, functions, or matrices).

### Column Vector 
An **n-dimensional column vector** in $ \mathbb{R}^n $ is represented as:

$$

\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n

$$

where each $$ v_i $$ is a real number. Using transpose notation, this can also be written as the transpose of a row vector:

$$

\mathbf{v} = (v_1, v_2, \dots, v_n)^\top

$$

### Row Vector 
An **n-dimensional row vector** in $$ \mathbb{R}^n $$ is represented as:

$$

\mathbf{v} = \begin{bmatrix} v_1 & v_2 & \dots & v_n \end{bmatrix} \in \mathbb{R}^n

$$

Alternatively, it can be written in tuple form:

$$

\mathbf{v} = (v_1, v_2, \dots, v_n)

$$

The column vector form is the transpose of the row vector, i.e., $$ \mathbf{v}^\top = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} $$.


## Vector Operations

Given two column vectors $ \mathbf{u} $ and $ \mathbf{v} $, the following operations can be performed:

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

**Example: 3D Vectors**
Consider three vectors in $$ \mathbb{R}^3 $$:

$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad 
\mathbf{v}_2 = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}, \quad 
\mathbf{v}_3 = \begin{bmatrix} 7 \\ 8 \\ 9 \end{bmatrix}
$$

A general linear combination of these vectors is:

$$
\mathbf{w} = c_1 \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + 
c_2 \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} + 
c_3 \begin{bmatrix} 7 \\ 8 \\ 9 \end{bmatrix}
$$

$$
= \begin{bmatrix} c_1 + 4c_2 + 7c_3 \\ 2c_1 + 5c_2 + 8c_3 \\ 3c_1 + 6c_2 + 9c_3 \end{bmatrix}
$$


## Magnitude of a Vector
Before that, lets see the definition of norm. Norm is represented as $$ \|.\| $$
### Properties of a Norm

For a vector norm $$ \|\mathbf{v}\| $$, the following properties hold:

1. **Non-negativity**: $$ \|\mathbf{v}\|  \geq 0 $$, and $$ \|\mathbf{v}\| = 0 $$ if and only if $ \mathbf{v} = \mathbf{0} $.
2. **Absolute Scalability**: For any scalar $ \alpha $,

$$
\|\alpha \mathbf{v}\| = |\alpha| \|\mathbf{v}\|
$$

3. **Triangle Inequality**: $$ \|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\| $$.
4. **Norm Definiteness**: If $$ \|\mathbf{v}\| = 0 $$, then $ \mathbf{v} $ must be the zero vector.

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

As $ v_i^2 \geq 0 $ for all $ i $, it follows that $$ \|\mathbf{v}\|_2 \geq 0 $$. If $$ \|\mathbf{v}\|_2 = 0 $$, then:

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
Follows from non-negativity: $$ \|\mathbf{v}\|_2 = 0 $$ if and only if $ \mathbf{v} = \mathbf{0} $.

## Homework: Verify Other Norms

Now that we’ve proved the Euclidean norm (L2 norm) is valid, let’s explore other common norms. Try proving these yourself!

### L1 Norm
The L1 norm (or Manhattan norm) of a vector $$ \mathbf{v} = [v_1, v_2, \dots, v_n] $$ is:

$$
\|\mathbf{v}\|_1 = |v_1| + |v_2| + \dots + |v_n|
$$

**Exercise**: Verify that $$ \|\mathbf{v}\|_1 $$ satisfies all the properties of norm or not?
*Hint*: Use the triangle inequality for real numbers ($$ |a + b| \leq |a| + |b| $$) for property 3.

### L∞ Norm
The L∞ norm (or maximum norm) of a vector $ \mathbf{v} = [v_1, v_2, \dots, v_n] $ is:

$$
\|\mathbf{v}\|_\infty = \max_{i} |v_i|
$$



**Exercise**: Verify that $$ \|\mathbf{v}\|_\infty $$ satisfies all the properties of norm or not?
*Hint*: Consider how the maximum of a sum relates to the sum of maximums for property 3.

## Challenge: Is the Minimum Norm Valid?

Now, suppose we define a new "norm" by replacing $ \max $ with $ \min $ in the L∞ norm:

$$
\|\mathbf{v}\|_\text{min} = \min_{i} |v_i|
$$

**Question**: Does $$ \|\mathbf{v}\|_\text{min} $$ satisfy all four norm properties?
- Try a simple example, like $ \mathbf{u} = [1, 0] $ and $ \mathbf{v} = [0, 1] $, to check the triangle inequality.


## Dot Product of Two Vectors

The **dot product** (or scalar product) of two vectors $$ \mathbf{u} $$ and $$ \mathbf{v} $$ in $$ \mathbb{R}^n $$ is defined as:

$$

\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + \dots + u_n v_n

$$

For column vectors $$ \mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{bmatrix} $$ and $$ \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} $$, the dot product is a scalar given by the sum of the products of their corresponding components:

$$

\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i

$$

**Prove that $$ \boxed{\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta }$$**

Consider two vectors $$ \mathbf{u} = (u_1, \dots, u_n)^\top $$ and $$ \mathbf{v} = (v_1, \dots, v_n)^\top $$ in $$ \mathbb{R}^n $$, where $$ \theta $$ is the angle between them. We aim to show:

$$

\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos\theta

$$

#### Steps
- Use the law of cosines in the triangle formed by $$ \mathbf{u} $$, $$ \mathbf{v} $$, and $$ \mathbf{u} - \mathbf{v} $$:

$$

\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2 \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos\theta

$$

- Expand the left-hand side using the Euclidean norm:

$$

\begin{align*}
\|\mathbf{u} - \mathbf{v}\|_2^2 &= \sum_{i=1}^{n} (u_i - v_i)^2 \\
&= \sum_{i=1}^{n} (u_i^2 - 2 u_i v_i + v_i^2) \\
&= \sum_{i=1}^{n} u_i^2 - 2 \sum_{i=1}^{n} u_i v_i + \sum_{i=1}^{n} v_i^2 \\
&= \|\mathbf{u}\|_2^2 - 2 \mathbf{u} \cdot \mathbf{v} + \|\mathbf{v}\|_2^2
\end{align*}

$$

- Equate the two expressions:

$$

\begin{align*}
\|\mathbf{u}\|_2^2 - 2 \mathbf{u} \cdot \mathbf{v} + \|\mathbf{v}\|_2^2 &= \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2 \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos\theta \\
-2 \mathbf{u} \cdot \mathbf{v} &= -2 \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos\theta
\end{align*}

$$

$$

\boxed{\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos\theta}

$$

Hence proved
## Unit Vectors

A **unit vector** is a vector with a magnitude of 1, i.e., $$ \|\mathbf{u}\| = 1 $$. For any non-zero vector $$ \mathbf{v} $$, its unit vector in the same direction is:

$$

\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}

$$


----------------------------------------------
