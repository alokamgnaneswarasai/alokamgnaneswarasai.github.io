---
layout: post
title: Introduction to Multi-variate calculus
tags: [calculus,differentiation,minima,maxima,inflection]
author: Gnaneswara Sai
---



In the previous post we have discussed about the univariate calculus, if you are not familiar with it you can read here [Introduction to Univariate Calculus]({{ site.baseurl }}/Vectorcalculus/)




## From Univariate to Multivariate Calculus: Introducing Partial Differentiation

### Univariate Calculus: Functions from $\mathbb{R} \to \mathbb{R}$

- In univariate calculus, we deal with functions of a single variable, $f: \mathbb{R} \to \mathbb{R}$, where the input is a single real number $x$ and the output is a real number $f(x)$.
- The derivative, denoted $f'(x)$ or $\frac{df}{dx}$, measures the rate of change of $f(x)$ with respect to $x$.
- **Example**: For $f(x) = x^2$:
  - $$ f'(x) = \frac{d}{dx} (x^2) = 2x $$
- This tells us how $f(x)$ changes as $x$ varies, useful for optimization, slopes, and rates of change in one-dimensional problems.

### Multivariate Calculus:

- Now consider functions with multiple variables, such as $f: \mathbb{R}^2 \to \mathbb{R}$, where the input is a pair $(x, y)$ and the output is a single real number $f(x, y)$.
- Example: $f(x, y) = x^2 + y^2$ represents the height of a paraboloid at point $(x, y)$.
- With two variables, the rate of change of $f(x, y)$ depends on the direction: changing $x$ while keeping $y$ fixed, or vice versa. A single derivative no longer suffices because $f$ varies across a 2D plane, not just a line.

### Why Partial Differentiation is Needed

- Partial differentiation extends the concept of derivatives to multivariate functions by measuring the rate of change of $f(x, y)$ with respect to one variable while holding others constant.
- It’s essential for:
  - Optimization in higher dimensions (e.g., minimizing cost functions in machine learning with multiple parameters).
  - Understanding how each variable independently affects the output (e.g., in physics, how temperature changes with position).
- Notation: $\frac{\partial f}{\partial x}$ for the partial derivative with respect to $x$, and $\frac{\partial f}{\partial y}$ for $y$.

### Definition of Partial Differentiation in $n$-Dimensional Space

Let $f: \mathbb{R}^n \to \mathbb{R}$ be a function of $n$ variables, i.e.,  

$$
f(x_1, x_2, \dots, x_n)
$$  

The partial derivative of $f$ with respect to the variable $x_i$ at a point $(x_1, x_2, \dots, x_n)$ is defined as:  

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_i, \dots, x_n)}{h}
$$  

This measures the rate of change of $f$ with respect to $x_i$ while keeping all other variables $(x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n)$ fixed.  




### Introducing the Gradient

- The gradient of a function $f: \mathbb{R}^n \to \mathbb{R}$ is a vector that collects all partial derivatives, representing the direction and rate of steepest increase of $f$ at a point.

- There are two common layouts for the gradient vector:
  1. **Numerator Layout**: A row vector:  
     
     $$ \nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right) $$
  2. **Denominator Layout**: A column vector:  
    
     $$ \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix} $$
- **Note**: In the rest of this article, we will use the **numerator layout** (row vector) for consistency and simplicity.

### Example: Gradient of $f(x_1, x_2) = x_1^2 + x_2^2$

- **Problem Statement**: Compute the gradient of $f(x_1, x_2) = x_1^2 + x_2^2$ at the point $(1, 2)$ using the numerator layout.

- **Gradient Vector**:  
  
  $$ \nabla f_{x_1, x_2} = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2} \right] $$
  
   $$ \boxed{ \frac{\partial f}{\partial x_1} = \frac{\partial}{\partial x_1} (x_1^2 + x_2^2) = 2x_1 } $$

   $$ \boxed{ \frac{\partial f}{\partial x_2} = \frac{\partial}{\partial x_2} (x_1^2 + x_2^2) = 2x_2 } $$
   
   $$ \nabla f_{x_1, x_2} = \left[ 2x_1, 2x_2 \right] $$

   $$ \nabla f_{1, 2} = \left[ 2, 4 \right] $$


## Basic Rules of Partial Differentiation in the Multivariate Case

- In the multivariate case, where $x \in \mathbb{R}^n$, the basic differentiation rules (sum, product, chain) apply, but gradients involve vectors and matrices, and matrix multiplication is not commutative (order matters).

### Rules

1. **Sum Rule**:  
   
   $$ \frac{\partial}{\partial x} \left( f(x) + g(x) \right) = \frac{\partial f}{\partial x} + \frac{\partial g}{\partial x} $$


2. **Product Rule**:  
   
   $$ \frac{\partial}{\partial x} \left( f(x) g(x) \right) = \frac{\partial f}{\partial x} g(x) + f(x) \frac{\partial g}{\partial x} $$


3. **Chain Rule**:  
   
   $$ \frac{\partial}{\partial x} \left( g(f(x)) \right) = \frac{\partial g}{\partial f} \frac{\partial f}{\partial x} $$


### Why Order Matters

- For $x \in \mathbb{R}^n$, $\frac{\partial f}{\partial x}$ is a vector (gradient):  
  
  $$ \frac{\partial f}{\partial x} = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right) $$

- In the chain rule, $\frac{\partial g}{\partial f} \frac{\partial f}{\partial x}$ resembles matrix multiplication:
  - $\frac{\partial f}{\partial x}$ is a row vector (numerator layout).
  - $\frac{\partial g}{\partial f}$ must align dimensionally for multiplication to be defined.
- Matrix multiplication is not commutative ($AB \neq BA$), so the order of terms must be preserved to ensure correct computation.


## Gradients of Vector-Valued Functions

- Previously, we discussed gradients of scalar functions $f: \mathbb{R}^n \to \mathbb{R}$. Now, we generalize to vector-valued functions $f: \mathbb{R}^n \to \mathbb{R}^m$, where $n \geq 1$ and $m > 1$.
- For 
  $$x = \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$, 

  the function is:  

  $$ f(x) = \begin{bmatrix} f_1(x) \\ \vdots \\ f_m(x) \end{bmatrix} \in \mathbb{R}^m $$

- View $f$ as a vector of functions $[f_1, \dots, f_m]^\top$, where each $f_i: \mathbb{R}^n \to \mathbb{R}$ follows the differentiation rules from Section 5.2.

### Partial Derivative with Respect to $x_i$

- For $f: \mathbb{R}^n \to \mathbb{R}^m$, the partial derivative with respect to $x_i$ is:  
  $$ \frac{\partial f}{\partial x_i} = \begin{bmatrix} 
  \frac{\partial f_1}{\partial x_i} \\ 
  \vdots \\ 
  \frac{\partial f_m}{\partial x_i} 
  \end{bmatrix} = \begin{bmatrix} 
  \lim_{h \to 0} \frac{f_1(x_1, \dots, x_i + h, \dots, x_n) - f_1(x)}{h} \\ 
  \vdots \\ 
  \lim_{h \to 0} \frac{f_m(x_1, \dots, x_i + h, \dots, x_n) - f_m(x)}{h} 
  \end{bmatrix} \in \mathbb{R}^m $$

### Gradient (Jacobian) of $f: \mathbb{R}^n \to \mathbb{R}^m$

- The gradient with respect to $x$ is a matrix of all partial derivatives:  
  
  $$ \frac{df(x)}{dx} = \left[ \frac{\partial f(x)}{\partial x_1} \quad \cdots \quad \frac{\partial f(x)}{\partial x_n} \right] $$

  $$     = \begin{bmatrix} 
  \frac{\partial f_1(x)}{\partial x_1} & \cdots & \frac{\partial f_1(x)}{\partial x_n} \\ 
  \vdots & \ddots & \vdots \\ 
  \frac{\partial f_m(x)}{\partial x_1} & \cdots & \frac{\partial f_m(x)}{\partial x_n} 
  \end{bmatrix} \in \mathbb{R}^{m \times n} $$

- **Definition (Jacobian)**: The Jacobian $J = \nabla_x f = \frac{df(x)}{dx}$ is an $m \times n$ matrix:  
  
  $$ J = \begin{bmatrix} 
  \frac{\partial f_1(x)}{\partial x_1} & \cdots & \frac{\partial f_1(x)}{\partial x_n} \\ 
  \vdots & \ddots & \vdots \\ 
  \frac{\partial f_m(x)}{\partial x_1} & \cdots & \frac{\partial f_m(x)}{\partial x_n} 
  \end{bmatrix}, \quad J(i, j) = \frac{\partial f_i}{\partial x_j} $$

### Transition from Row Vector to Column Vectors in the Jacobian

The Jacobian matrix is initially represented as a row vector of partial derivatives:

$$\frac{df(x)}{dx} = \left[ \frac{\partial f(x)}{\partial x_1} \quad \frac{\partial f(x)}{\partial x_2} \quad \cdots \quad \frac{\partial f(x)}{\partial x_n} \right]$$

Each partial derivative in the row vector is replaced by a column vector of partial derivatives for each component of the vector-valued function $f(x)$:

$$\frac{\partial f(x)}{\partial x_1} = \begin{bmatrix} \frac{\partial f_1(x)}{\partial x_1} \\ \frac{\partial f_2(x)}{\partial x_1} \\ \vdots \\ \frac{\partial f_m(x)}{\partial x_1} \end{bmatrix}, \quad\frac{\partial f(x)}{\partial x_2} = \begin{bmatrix} \frac{\partial f_1(x)}{\partial x_2} \\ 
\frac{\partial f_2(x)}{\partial x_2} \\ 
\vdots \\ 
\frac{\partial f_m(x)}{\partial x_2} 
\end{bmatrix}, \quad \cdots, \quad
\frac{\partial f(x)}{\partial x_n} = \begin{bmatrix} 
\frac{\partial f_1(x)}{\partial x_n} \\ 
\frac{\partial f_2(x)}{\partial x_n} \\ 
\vdots \\ 
\frac{\partial f_m(x)}{\partial x_n} 
\end{bmatrix}
$$

Finally, these column vectors are combined to form the Jacobian matrix:

$$
\frac{df(x)}{dx} = \begin{bmatrix} 
\frac{\partial f_1(x)}{\partial x_1} & \frac{\partial f_1(x)}{\partial x_2} & \cdots & \frac{\partial f_1(x)}{\partial x_n} \\ 
\frac{\partial f_2(x)}{\partial x_1} & \frac{\partial f_2(x)}{\partial x_2} & \cdots & \frac{\partial f_2(x)}{\partial x_n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\frac{\partial f_m(x)}{\partial x_1} & \frac{\partial f_m(x)}{\partial x_2} & \cdots & \frac{\partial f_m(x)}{\partial x_n} 
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

This transition shows how the Jacobian matrix is constructed by replacing each partial derivative in the row vector with its corresponding column vector.

### Example: Jacobian of $f(x_1, x_2) = (x_1^2 x_2, x_2^2)$

- $f(x_1, x_2) = \begin{bmatrix} x_1^2 x_2 \\ x_2^2 \end{bmatrix}$, compute $J$ at $(1, 2)$:
  
  $$ \frac{\partial f_1}{\partial x_1} = 2x_1 x_2, \quad \frac{\partial f_1}{\partial x_2} = x_1^2 $$

   $$ \frac{\partial f_2}{\partial x_1} = 0, \quad \frac{\partial f_2}{\partial x_2} = 2x_2 $$

   $$ J = \nabla_x f = \begin{bmatrix} 
  2x_1 x_2 & x_1^2 \\ 
  0 & 2x_2 
  \end{bmatrix} $$

   At $(1, 2)$:  

    $$ J(1, 2) = \begin{bmatrix} 
    2 \cdot 1 \cdot 2 & 1^2 \\ 
    0 & 2 \cdot 2 
    \end{bmatrix} = \begin{bmatrix} 
    4 & 1 \\ 
    0 & 4 
    \end{bmatrix} $$

---------

Let $A$, $x$, and $f(x)$ be represented as follows:

1. **Matrix $A$:**
   
   $$
   A = \begin{bmatrix} 
   A_{11} & A_{12} & \cdots & A_{1N} \\ 
   A_{21} & A_{22} & \cdots & A_{2N} \\ 
   \vdots & \vdots & \ddots & \vdots \\ 
   A_{M1} & A_{M2} & \cdots & A_{MN} 
   \end{bmatrix}
   $$

2. **Vector $x$:**
   
   $$
   x = \begin{bmatrix} 
   x_1 \\ 
   x_2 \\ 
   \vdots \\ 
   x_N 
   \end{bmatrix}
   $$

3. **Result $f(x) = Ax$:**
   
   $$
   f(x) = \begin{bmatrix} 
   f_1(x) \\ 
   f_2(x) \\ 
   \vdots \\ 
   f_M(x) 
   \end{bmatrix} = \begin{bmatrix} 
   A_{11} x_1 + A_{12} x_2 + \cdots + A_{1N} x_N \\ 
   A_{21} x_1 + A_{22} x_2 + \cdots + A_{2N} x_N \\ 
   \vdots \\ 
   A_{M1} x_1 + A_{M2} x_2 + \cdots + A_{MN} x_N 
   \end{bmatrix}
   $$

---

### Proof: $ \frac{\partial}{\partial x}(Ax) = A $



<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      Click to view the proof
    </summary>
    <div style="margin-top: 5px;">
      Let $ A \in \mathbb{R}^{m \times n} $ and $ x \in \mathbb{R}^n $. The function $ f(x) = Ax $ is defined as:

        $$
        f(x) = \begin{bmatrix} 
        f_1(x) \\ 
        f_2(x) \\ 
        \vdots \\ 
        f_m(x) 
        \end{bmatrix} = \begin{bmatrix} 
        \sum_{j=1}^n A_{1j} x_j \\ 
        \sum_{j=1}^n A_{2j} x_j \\ 
        \vdots \\ 
        \sum_{j=1}^n A_{mj} x_j 
        \end{bmatrix}
        $$
      The partial derivative of \( f_i(x) \) with respect to \( x_k \) is:

      $$
      \frac{\partial f_i}{\partial x_k} = \frac{\partial}{\partial x_k} \left( \sum_{j=1}^n A_{ij} x_j \right)
      $$

      Since \( A_{ij} \) is constant, the derivative simplifies to:

      $$
      \frac{\partial f_i}{\partial x_k} = A_{ik}
      $$

       Jacobian Construction
      Arrange the partial derivatives \( \frac{\partial f_i}{\partial x_k} \) into a matrix:

      $$
      \frac{\partial f}{\partial x} = \begin{bmatrix} 
      \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ 
      \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} 
      \end{bmatrix}
      $$

      Substituting \( \frac{\partial f_i}{\partial x_k} = A_{ik} \), we get:

      $$
      \frac{\partial f}{\partial x} = A
      $$

      ---
       Final Result
      The derivative of \( f(x) = Ax \) is:

      $$
      \boxed{\frac{\partial}{\partial x}(Ax) = A}
      $$
    </div>
  </details>
</div>

---

### Proof: $\frac{\partial}{\partial x} (\mathbf{u} \cdot \mathbf{v}) = \mathbf{u}^\top \frac{\partial \mathbf{v}}{\partial x} + \mathbf{v}^\top \frac{\partial \mathbf{u}}{\partial x}$ where  $ \mathbf{u} = \mathbf{u}(x) \in \mathbb{R}^n $ and $ \mathbf{v} = \mathbf{v}(x) \in \mathbb{R}^n $



<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      Click to view the proof
    </summary>
    <div style="margin-top: 5px;">
      Let $ \mathbf{u} = \mathbf{u}(x) \in \mathbb{R}^n $ and $ \mathbf{v} = \mathbf{v}(x) \in \mathbb{R}^n $. The dot product $ \mathbf{u} \cdot \mathbf{v} $ is defined as:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
$$
      The derivative of \( \mathbf{u} \cdot \mathbf{v} \) with respect to \( x \) is:

      $$
      \frac{\partial}{\partial x} (\mathbf{u} \cdot \mathbf{v}) = \frac{\partial}{\partial x} \left( \sum_{i=1}^n u_i v_i \right)
      $$

      Using the product rule for each term \( u_i v_i \):

      $$
      \frac{\partial}{\partial x} (\mathbf{u} \cdot \mathbf{v}) = \sum_{i=1}^n \left( \frac{\partial u_i}{\partial x} v_i + u_i \frac{\partial v_i}{\partial x} \right)
      $$

  

      

      $$
      \frac{\partial}{\partial x} (\mathbf{u} \cdot \mathbf{v}) = \mathbf{u}^\top \frac{\partial \mathbf{v}}{\partial x} + \mathbf{v}^\top \frac{\partial \mathbf{u}}{\partial x}
      $$

      
      
       Final Result
      The derivative of \( \mathbf{u} \cdot \mathbf{v} \) is:

      $$
      \boxed{\frac{\partial}{\partial x} (\mathbf{u} \cdot \mathbf{v}) = \mathbf{u}^\top \frac{\partial \mathbf{v}}{\partial x} + \mathbf{v}^\top \frac{\partial \mathbf{u}}{\partial x}}
      $$
    </div>
  </details>
</div>