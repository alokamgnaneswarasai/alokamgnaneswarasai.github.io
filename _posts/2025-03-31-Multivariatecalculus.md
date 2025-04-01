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

- View $f$ as a vector of functions $[f_1, \dots, f_m]^\top$, where each $f_i: \mathbb{R}^n \to \mathbb{R}$ follows the differentiation rules as mentioned above.

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


  <!-- $$     = \begin{bmatrix} 
  \frac{\partial f_1(x)}{\partial x_1} & \cdots & \frac{\partial f_1(x)}{\partial x_n} \\ 
  \vdots & \ddots & \vdots \\ 
  \frac{\partial f_m(x)}{\partial x_1} & \cdots & \frac{\partial f_m(x)}{\partial x_n} 
  \end{bmatrix} \in \mathbb{R}^{m \times n} $$ -->

- **Definition (Jacobian)**: The Jacobian $J = \nabla_x f = \frac{df(x)}{dx}$ is an $m \times n$ matrix:  
  
  $$ J = \begin{bmatrix} 
  \frac{\partial f_1(x)}{\partial x_1} & \cdots & \frac{\partial f_1(x)}{\partial x_n} \\ 
  \vdots & \ddots & \vdots \\ 
  \frac{\partial f_m(x)}{\partial x_1} & \cdots & \frac{\partial f_m(x)}{\partial x_n} 
  \end{bmatrix}, \quad J(i, j) = \frac{\partial f_i}{\partial x_j} $$



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
 **There are two layout notations , some authors prefer numerator layout and some authors prefer denominator layout. In this post we are using numerator layout**

## Numerator-Layout Notation

- In numerator-layout notation, the gradient aligns with the numerator’s structure:
  - For scalar $y$ and vector $x \in \mathbb{R}^n$:

    $$ \frac{\partial y}{\partial x} = \begin{bmatrix} \frac{\partial y}{\partial x_1} & \frac{\partial y}{\partial x_2} & \cdots & \frac{\partial y}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{1 \times n} $$

  - For vector $y \in \mathbb{R}^m$ and scalar $x$:
    
    $$ \frac{\partial y}{\partial x} = \begin{bmatrix} \frac{\partial y_1}{\partial x} \\ \frac{\partial y_2}{\partial x} \\ \vdots \\ \frac{\partial y_m}{\partial x} \end{bmatrix} \in \mathbb{R}^{m \times 1} $$

  - For vector $y \in \mathbb{R}^m$ and vector $x \in \mathbb{R}^n$:
    
    $$ \frac{\partial y}{\partial x} = \begin{bmatrix} 
    \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_n} \\ 
    \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_n} \\ 
    \vdots & \vdots & \ddots & \vdots \\ 
    \frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_n} 
    \end{bmatrix} \in \mathbb{R}^{m \times n} $$

  - For scalar $y$ and matrix $X \in \mathbb{R}^{p \times q}$:
   
    $$ \frac{\partial y}{\partial X} = \begin{bmatrix} 
    \frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{21}} & \cdots & \frac{\partial y}{\partial x_{p1}} \\ 
    \frac{\partial y}{\partial x_{12}} & \frac{\partial y}{\partial x_{22}} & \cdots & \frac{\partial y}{\partial x_{p2}} \\ 
    \vdots & \vdots & \ddots & \vdots \\ 
    \frac{\partial y}{\partial x_{1q}} & \frac{\partial y}{\partial x_{2q}} & \cdots & \frac{\partial y}{\partial x_{pq}} 
    \end{bmatrix} \in \mathbb{R}^{p \times q} $$

  - For matrix $Y \in \mathbb{R}^{m \times n}$ and scalar $x$:
    
    $$ \frac{\partial Y}{\partial x} = \begin{bmatrix} 
    \frac{\partial y_{11}}{\partial x} & \frac{\partial y_{12}}{\partial x} & \cdots & \frac{\partial y_{1n}}{\partial x} \\ 
    \frac{\partial y_{21}}{\partial x} & \frac{\partial y_{22}}{\partial x} & \cdots & \frac{\partial y_{2n}}{\partial x} \\ 
    \vdots & \vdots & \ddots & \vdots \\ 
    \frac{\partial y_{m1}}{\partial x} & \frac{\partial y_{m2}}{\partial x} & \cdots & \frac{\partial y_{mn}}{\partial x} 
    \end{bmatrix} \in \mathbb{R}^{m \times n} $$

  - Differential $dX$ for $X \in \mathbb{R}^{m \times n}$:
    
    $$ dX = \begin{bmatrix} 
    dx_{11} & dx_{12} & \cdots & dx_{1n} \\ 
    dx_{21} & dx_{22} & \cdots & dx_{2n} \\ 
    \vdots & \vdots & \ddots & \vdots \\ 
    dx_{m1} & dx_{m2} & \cdots & dx_{mn} 
    \end{bmatrix} \in \mathbb{R}^{m \times n} $$

## Denominator-Layout Notation

- In denominator-layout notation, the gradient aligns with the denominator’s structure:
  - For scalar $y$ and vector $x \in \mathbb{R}^n$:
    
    $$ \frac{\partial y}{\partial x} = \begin{bmatrix} \frac{\partial y}{\partial x_1} \\ \frac{\partial y}{\partial x_2} \\ \vdots \\ \frac{\partial y}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{n \times 1} $$

  - For vector $y \in \mathbb{R}^m$ and scalar $x$:
  
    $$ \frac{\partial y}{\partial x} = \begin{bmatrix} \frac{\partial y_1}{\partial x} & \frac{\partial y_2}{\partial x} & \cdots & \frac{\partial y_m}{\partial x} \end{bmatrix} \in \mathbb{R}^{1 \times m} $$

  - For vector $y \in \mathbb{R}^m$ and vector $x \in \mathbb{R}^n$:
    
    $$ \frac{\partial y}{\partial x} = \begin{bmatrix} 
    \frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_1} \\ 
    \frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_2} \\ 
    \vdots & \vdots & \ddots & \vdots \\ 
    \frac{\partial y_1}{\partial x_n} & \frac{\partial y_2}{\partial x_n} & \cdots & \frac{\partial y_m}{\partial x_n} 
    \end{bmatrix} \in \mathbb{R}^{n \times m} $$

  - For scalar $y$ and matrix $X \in \mathbb{R}^{p \times q}$:
   
    $$ \frac{\partial y}{\partial X} = \begin{bmatrix} 
    \frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{12}} & \cdots & \frac{\partial y}{\partial x_{1q}} \\ 
    \frac{\partial y}{\partial x_{21}} & \frac{\partial y}{\partial x_{22}} & \cdots & \frac{\partial y}{\partial x_{2q}} \\ 
    \vdots & \vdots & \ddots & \vdots \\ 
    \frac{\partial y}{\partial x_{p1}} & \frac{\partial y}{\partial x_{p2}} & \cdots & \frac{\partial y}{\partial x_{pq}} 
    \end{bmatrix} \in \mathbb{R}^{p \times q} $$


<!-- Let $A$, $x$, and $f(x)$ be represented as follows:

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
   $$ -->

---
## Vecor by Vector

For vector $y \in \mathbb{R}^m$ and vector $x \in \mathbb{R}^n$:
    
$$ \frac{\partial y}{\partial x} = \begin{bmatrix} 
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_n} \\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_n} 
\end{bmatrix} \in \mathbb{R}^{m \times n} $$

<div id="proof-derivative-Ax" class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{\partial}{\partial x}(Ax) = A $$
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



<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial (a \mathbf{u})}{\partial \mathbf{x}} = a \frac{\partial \mathbf{u}}{\partial \mathbf{x}}$$
    </summary>
    <div style="margin-top: 10px;">

      Let \( a \) be a constant scalar, and \( \mathbf{u} = \mathbf{u}(\mathbf{x}) \in \mathbb{R}^N \) be a vector-valued function of \( \mathbf{x} \in \mathbb{R}^M \).  
      The function \( f(\mathbf{x}) = a \mathbf{u}(\mathbf{x}) \) scales the vector \( \mathbf{u} \) by the constant \( a \).

      
      The derivative of \( f(\mathbf{x}) = a \mathbf{u}(\mathbf{x}) \) with respect to \( \mathbf{x} \) is defined as:

      $$
      \frac{\partial (a \mathbf{u})}{\partial \mathbf{x}} = \begin{bmatrix} 
      \frac{\partial (a u_1)}{\partial x_1} & \cdots & \frac{\partial (a u_1)}{\partial x_M} \\ 
      \vdots & \ddots & \vdots \\ 
      \frac{\partial (a u_N)}{\partial x_1} & \cdots & \frac{\partial (a u_N)}{\partial x_M} 
      \end{bmatrix}
      $$

      
      Since \( a \) is constant, it can be factored out of the derivative:

      $$
      \frac{\partial (a u_i)}{\partial x_j} = a \frac{\partial u_i}{\partial x_j}
      $$

      Substituting this into the matrix form:

      $$
      \frac{\partial (a \mathbf{u})}{\partial \mathbf{x}} = a \begin{bmatrix} 
      \frac{\partial u_1}{\partial x_1} & \cdots & \frac{\partial u_1}{\partial x_M} \\ 
      \vdots & \ddots & \vdots \\ 
      \frac{\partial u_N}{\partial x_1} & \cdots & \frac{\partial u_N}{\partial x_M} 
      \end{bmatrix}
      $$

      
      we can observe  that the matrix of partial derivatives is \( \frac{\partial \mathbf{u}}{\partial \mathbf{x}} \), we have:

      $$
      \frac{\partial (a \mathbf{u})}{\partial \mathbf{x}} = a \frac{\partial \mathbf{u}}{\partial \mathbf{x}}
      $$

       Final Result
      The derivative of \( a \mathbf{u} \) with respect to \( \mathbf{x} \) is:

      $$
      \boxed{\frac{\partial (a \mathbf{u})}{\partial \mathbf{x}} = a \frac{\partial \mathbf{u}}{\partial \mathbf{x}}}
      $$

    </div>
  </details>
</div>

<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial \mathbf{a}}{\partial \mathbf{x}} = \mathbf{0}, \quad \frac{\partial \mathbf{x}}{\partial \mathbf{x}} = \mathbf{I}$$
    </summary>
    <div style="margin-top: 10px;">

       Proof 1: \( \frac{\partial \mathbf{a}}{\partial \mathbf{x}} = \mathbf{0} \)
      Let \( \mathbf{a} \in \mathbb{R}^N \) be a constant vector, and \( \mathbf{x} \in \mathbb{R}^M \) be the variable vector.  
      The derivative of \( \mathbf{a} \) with respect to \( \mathbf{x} \) is defined as:

      $$
      \frac{\partial \mathbf{a}}{\partial \mathbf{x}} = \begin{bmatrix} 
      \frac{\partial a_1}{\partial x_1} & \cdots & \frac{\partial a_1}{\partial x_M} \\ 
      \vdots & \ddots & \vdots \\ 
      \frac{\partial a_N}{\partial x_1} & \cdots & \frac{\partial a_N}{\partial x_M} 
      \end{bmatrix}
      $$

      Since \( \mathbf{a} \) is constant, each component \( a_i \) does not depend on \( \mathbf{x} \). Therefore, all partial derivatives are zero:

      $$
      \frac{\partial a_i}{\partial x_j} = 0 \quad \forall i, j
      $$

      Substituting this into the matrix form:

      $$
      \frac{\partial \mathbf{a}}{\partial \mathbf{x}} = \begin{bmatrix} 
      0 & \cdots & 0 \\ 
      \vdots & \ddots & \vdots \\ 
      0 & \cdots & 0 
      \end{bmatrix} = \mathbf{0}
      $$

       Proof 2: \( \frac{\partial \mathbf{x}}{\partial \mathbf{x}} = \mathbf{I} \)
      Let \( \mathbf{x} \in \mathbb{R}^N \) be the variable vector. The derivative of \( \mathbf{x} \) with respect to itself is defined as:

      $$
      \frac{\partial \mathbf{x}}{\partial \mathbf{x}} = \begin{bmatrix} 
      \frac{\partial x_1}{\partial x_1} & \cdots & \frac{\partial x_1}{\partial x_N} \\ 
      \vdots & \ddots & \vdots \\ 
      \frac{\partial x_N}{\partial x_1} & \cdots & \frac{\partial x_N}{\partial x_N} 
      \end{bmatrix}
      $$

      The partial derivative \( \frac{\partial x_i}{\partial x_j} \) is:

      $$
      \frac{\partial x_i}{\partial x_j} = 
      \begin{cases} 
      1 & \text{if } i = j \\ 
      0 & \text{if } i \neq j 
      \end{cases}
      $$

      Substituting this into the matrix form, we get the identity matrix:

      $$
      \frac{\partial \mathbf{x}}{\partial \mathbf{x}} = \begin{bmatrix} 
      1 & 0 & \cdots & 0 \\ 
      0 & 1 & \cdots & 0 \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      0 & 0 & \cdots & 1 
      \end{bmatrix} = \mathbf{I}
      $$

       Final Results
      1. For a constant vector \( \mathbf{a} \), the derivative is:

      $$
      \boxed{\frac{\partial \mathbf{a}}{\partial \mathbf{x}} = \mathbf{0}}
      $$

      2. For the variable vector \( \mathbf{x} \), the derivative with respect to itself is:

      $$
      \boxed{\frac{\partial \mathbf{x}}{\partial \mathbf{x}} = \mathbf{I}}
      $$

    </div>
  </details>
</div>

---

## Scalar by vector 

For scalar $y$ and vector $x \in \mathbb{R}^n$:

$$ \frac{\partial y}{\partial x} = \begin{bmatrix} \frac{\partial y}{\partial x_1} & \frac{\partial y}{\partial x_2} & \cdots & \frac{\partial y}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{1 \times n} $$


<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial}{\partial x} (\mathbf{u} \cdot \mathbf{v}) = \mathbf{u}^\top \frac{\partial \mathbf{v}}{\partial x} + \mathbf{v}^\top \frac{\partial \mathbf{u}}{\partial x} $$
    </summary>
    <div style="margin-top: 10px;">

      Let \( \mathbf{u} = \mathbf{u}(x) \in \mathbb{R}^n \) and \( \mathbf{v} = \mathbf{v}(x) \in \mathbb{R}^n \). The dot product \( \mathbf{u} \cdot \mathbf{v} \) is defined as:

      $$
      \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
      $$

    Step 1: 
      The derivative of \( \mathbf{u} \cdot \mathbf{v} \) with respect to \( x \) is:

      $$
      \frac{\partial}{\partial x} (\mathbf{u} \cdot \mathbf{v}) = \frac{\partial}{\partial x} \left( \sum_{i=1}^n u_i v_i \right)
      $$

      Using the product rule for each term \( u_i v_i \):

      $$
      \frac{\partial}{\partial x} (\mathbf{u} \cdot \mathbf{v}) = \sum_{i=1}^n \left( \frac{\partial u_i}{\partial x} v_i + u_i \frac{\partial v_i}{\partial x} \right)
      $$

      $$
      \frac{\partial}{\partial x} (u \cdot v) = \sum_{i=1}^N \left( \frac{\partial u_i}{\partial x} v_i + u_i \frac{\partial v_i}{\partial x} \right)
        $$

        $$
        
        =
        \begin{bmatrix} 
        \sum_{i=1}^N \left( \frac{\partial u_i}{\partial x_1} v_i + u_i \frac{\partial v_i}{\partial x_1} \right) , & 
        \sum_{i=1}^N \left( \frac{\partial u_i}{\partial x_2} v_i + u_i \frac{\partial v_i}{\partial x_2} \right) , & 
        \cdots , & 
        \sum_{i=1}^N \left( \frac{\partial u_i}{\partial x_M} v_i + u_i \frac{\partial v_i}{\partial x_M} \right) 
        \end{bmatrix}
        
        $$

        $$
        
        \begin{bmatrix} 
        \sum_{i=1}^N v_i \frac{\partial u_i}{\partial x_1} + \sum_{i=1}^N u_i \frac{\partial v_i}{\partial x_1} , & 
        \cdots , & 
        \sum_{i=1}^N v_i \frac{\partial u_i}{\partial x_M} + \sum_{i=1}^N u_i \frac{\partial v_i}{\partial x_M} 
        \end{bmatrix}
        
        $$
        $$
        = \begin{bmatrix} v_1 & \cdots & v_N \end{bmatrix} \begin{bmatrix} \frac{\partial u_1}{\partial x_1} & \cdots & \frac{\partial u_1}{\partial x_M} \\ \vdots & \ddots & \vdots \\ \frac{\partial u_N}{\partial x_1} & \cdots & \frac{\partial u_N}{\partial x_M} \end{bmatrix} + \begin{bmatrix} u_1 & \cdots & u_N \end{bmatrix} \begin{bmatrix} \frac{\partial v_1}{\partial x_1} & \cdots & \frac{\partial v_1}{\partial x_M} \\ \vdots & \ddots & \vdots \\ \frac{\partial v_N}{\partial x_1} & \cdots & \frac{\partial v_N}{\partial x_M} \end{bmatrix}
        $$

        $$
        = v^\top \frac{\partial u}{\partial x} + u^\top \frac{\partial v}{\partial x}
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


<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial}{\partial x} \left( x^\top A x \right) = x^\top (A + A^\top)$$
    </summary>
    <div style="margin-top: 10px;">

      Let $ x = \begin{bmatrix} x_1 \\ \vdots \\ x_N \end{bmatrix} \in \mathbb{R}^N $, $ A \in \mathbb{R}^{N \times N} $, and $ f(x) = x^\top A x $.  
      $$ $$
      The function $ f(x) $ can be rewritten as dot product of x,Ax:

      $$
      f(x) = x^\top A x = x \cdot (A x)    
      $$

      Define $ u(x) = x $ and $ v(x) = A x $. Using the product rule for derivatives:

      $$
      \frac{\partial}{\partial x} (u \cdot v) = u^\top \frac{\partial v}{\partial x} + v^\top \frac{\partial u}{\partial x}
      $$

      Compute $ \frac{\partial u}{\partial x} $

      Since $ u(x) = x $, the derivative is:

      $$
      \frac{\partial u}{\partial x} = \frac{\partial x}{\partial x} = I_N
      $$

      where $ I_N $ is the $ N \times N $ identity matrix.

    $$  $$
    Compute $ \frac{\partial v}{\partial x} $

      Since $ v(x) = A x $, and we already proved that :

      $$
      \frac{\partial v}{\partial x} = A
      $$

        Substituting $ u = x $, $ v = A x $, $ \frac{\partial u}{\partial x} = I_N $, and $ \frac{\partial v}{\partial x} = A $ 

      $$
      \frac{\partial}{\partial x} \left( x^\top A x \right) = x^\top \frac{\partial (A x)}{\partial x} + (A x)^\top \frac{\partial x}{\partial x}
      $$

      Simplify each term:

      $$
      \frac{\partial}{\partial x} \left( x^\top A x \right) = x^\top A + (A x)^\top I_N
      $$

      Since $ (A x)^\top = x^\top A^\top $, we have :

      $$
      \frac{\partial}{\partial x} \left( x^\top A x \right) = x^\top A + x^\top A^\top
      $$

      
      Combine the terms $ A $ and $ A^\top $:

      $$
      \frac{\partial}{\partial x} \left( x^\top A x \right) = x^\top (A + A^\top)
      $$

    Final Result
      If $ A $ is symmetric ($ A^\top = A $), the result simplifies to:

      $$
      \frac{\partial}{\partial x} \left( x^\top A x \right) = 2 x^\top A
      $$

      Otherwise, the general result is:

      $$
      \boxed{\frac{\partial}{\partial x} \left( x^\top A x \right) = x^\top (A + A^\top)}
      $$

    </div>
  </details>
</div>
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial}{\partial x} \left( \mathbf{u}^\top A \mathbf{v} \right) = \mathbf{u}^\top A \frac{\partial \mathbf{v}}{\partial x} + \mathbf{v}^\top A^\top \frac{\partial \mathbf{u}}{\partial x}$$
    </summary>
    <div style="margin-top: 10px;">

      Let \( \mathbf{u} = \mathbf{u}(x) \in \mathbb{R}^N \), \( \mathbf{v} = \mathbf{v}(x) \in \mathbb{R}^N \), and \( A \in \mathbb{R}^{N \times N} \) be a constant matrix.  
      The function \( f(x) = \mathbf{u}^\top A \mathbf{v} \) can be rewritten as:

      $$
      f(x) = \mathbf{u} \cdot (A \mathbf{v})
      $$

      Using the identity:

      $$
      \frac{\partial}{\partial x} (\mathbf{u} \cdot \mathbf{v}) = \mathbf{u}^\top \frac{\partial \mathbf{v}}{\partial x} + \mathbf{v}^\top \frac{\partial \mathbf{u}}{\partial x}
      $$

      
      Here, \( \mathbf{u} \) remains as is, and \( \mathbf{v} \) is replaced by \( A \mathbf{v} \). Applying the identity:

      $$
      \frac{\partial}{\partial x} \left( \mathbf{u}^\top A \mathbf{v} \right) = \mathbf{u}^\top \frac{\partial (A \mathbf{v})}{\partial x} + (A \mathbf{v})^\top \frac{\partial \mathbf{u}}{\partial x}
      $$

       Compute \( \frac{\partial (A \mathbf{v})}{\partial x} \)
      Since \( A \) is constant, the derivative is:

      $$
      \frac{\partial (A \mathbf{v})}{\partial x} = A \frac{\partial \mathbf{v}}{\partial x}
      $$

       Simplify \( (A \mathbf{v})^\top \)
      The transpose of \( A \mathbf{v} \) is:

      $$
      (A \mathbf{v})^\top = \mathbf{v}^\top A^\top
      $$

      
      Substituting these results into the equation:

      $$
      \frac{\partial}{\partial x} \left( \mathbf{u}^\top A \mathbf{v} \right) = \mathbf{u}^\top A \frac{\partial \mathbf{v}}{\partial x} + \mathbf{v}^\top A^\top \frac{\partial \mathbf{u}}{\partial x}
      $$

      Final Result
      The derivative of \( \mathbf{u}^\top A \mathbf{v} \) is:

      $$
      \boxed{\frac{\partial}{\partial x} \left( \mathbf{u}^\top A \mathbf{v} \right) = \mathbf{u}^\top A \frac{\partial \mathbf{v}}{\partial x} + \mathbf{v}^\top A^\top \frac{\partial \mathbf{u}}{\partial x}}
      $$

    </div>
  </details>
</div>

-----


## Scalar-by-Matrix Differentiation

The derivative of a scalar function $ y $, with respect to a $ p \times q $ matrix $ \mathbf{X} $ of independent variables, is given (in **numerator layout notation**) by:

$$
\frac{\partial y}{\partial \mathbf{X}} = 
\begin{bmatrix} 
\frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{21}} & \cdots & \frac{\partial y}{\partial x_{p1}} \\ 
\frac{\partial y}{\partial x_{12}} & \frac{\partial y}{\partial x_{22}} & \cdots & \frac{\partial y}{\partial x_{p2}} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\frac{\partial y}{\partial x_{1q}} & \frac{\partial y}{\partial x_{2q}} & \cdots & \frac{\partial y}{\partial x_{pq}} 
\end{bmatrix}.
$$

### Explanation
- $ y $ is a scalar function of the elements of the matrix $ \mathbf{X} \in \mathbb{R}^{p \times q} $.
- Each element $ x_{ij} $ of $ \mathbf{X} $ is treated as an independent variable.
- The derivative $ \frac{\partial y}{\partial \mathbf{X}} $ is a $ p \times q $ matrix, where the $(i, j)$-th entry is the partial derivative of $ y $ with respect to $ x_{ij} $.

### Example
If $ y = \text{tr}(\mathbf{X}) = \sum_{i=1}^p x_{ii} $, then:

$$
\frac{\partial y}{\partial \mathbf{X}} = 
\begin{bmatrix} 
1 & 0 & \cdots & 0 \\ 
0 & 1 & \cdots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
0 & 0 & \cdots & 1 
\end{bmatrix}.
$$

This is the identity matrix because the trace depends only on the diagonal elements of $ \mathbf{X} $.

<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial (uv)}{\partial \mathbf{X}} = u \frac{\partial v}{\partial \mathbf{X}} + v \frac{\partial u}{\partial \mathbf{X}}$$
    </summary>
    <div style="margin-top: 10px;">

      
      Let \( u = u(\mathbf{X}) \in \mathbb{R} \) and \( v = v(\mathbf{X}) \in \mathbb{R} \) be scalar functions of the matrix \( \mathbf{X} \in \mathbb{R}^{p \times q} \).  
      The product \( uv \) is a scalar, and its derivative with respect to \( \mathbf{X} \) is defined as:

      $$
      \frac{\partial (uv)}{\partial \mathbf{X}} = 
      \begin{bmatrix} 
      \frac{\partial (uv)}{\partial x_{11}} & \frac{\partial (uv)}{\partial x_{21}} & \cdots & \frac{\partial (uv)}{\partial x_{p1}} \\ 
      \frac{\partial (uv)}{\partial x_{12}} & \frac{\partial (uv)}{\partial x_{22}} & \cdots & \frac{\partial (uv)}{\partial x_{p2}} \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      \frac{\partial (uv)}{\partial x_{1q}} & \frac{\partial (uv)}{\partial x_{2q}} & \cdots & \frac{\partial (uv)}{\partial x_{pq}} 
      \end{bmatrix}.
      $$

      
      Using the product rule for the scalar product \( uv \), the derivative of \( uv \) with respect to each element \( x_{ij} \) of \( \mathbf{X} \) is:

      $$
      \frac{\partial (uv)}{\partial x_{ij}} = u \frac{\partial v}{\partial x_{ij}} + v \frac{\partial u}{\partial x_{ij}}.
      $$

      

    Using the formula,

      $$
      \frac{\partial y}{\partial \mathbf{X}} = 
      \begin{bmatrix} 
      \frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{21}} & \cdots & \frac{\partial y}{\partial x_{p1}} \\ 
      \frac{\partial y}{\partial x_{12}} & \frac{\partial y}{\partial x_{22}} & \cdots & \frac{\partial y}{\partial x_{p2}} \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      \frac{\partial y}{\partial x_{1q}} & \frac{\partial y}{\partial x_{2q}} & \cdots & \frac{\partial y}{\partial x_{pq}} 
      \end{bmatrix},
      $$

      we can write the derivatives \( \frac{\partial v}{\partial \mathbf{X}} \) and \( \frac{\partial u}{\partial \mathbf{X}} \) in the same form. Substituting these into the product rule gives:

      $$
      \frac{\partial (uv)}{\partial \mathbf{X}} = 
      u 
      \begin{bmatrix} 
      \frac{\partial v}{\partial x_{11}} & \frac{\partial v}{\partial x_{21}} & \cdots & \frac{\partial v}{\partial x_{p1}} \\ 
      \frac{\partial v}{\partial x_{12}} & \frac{\partial v}{\partial x_{22}} & \cdots & \frac{\partial v}{\partial x_{p2}} \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      \frac{\partial v}{\partial x_{1q}} & \frac{\partial v}{\partial x_{2q}} & \cdots & \frac{\partial v}{\partial x_{pq}} 
      \end{bmatrix}
      +
      v 
      \begin{bmatrix} 
      \frac{\partial u}{\partial x_{11}} & \frac{\partial u}{\partial x_{21}} & \cdots & \frac{\partial u}{\partial x_{p1}} \\ 
      \frac{\partial u}{\partial x_{12}} & \frac{\partial u}{\partial x_{22}} & \cdots & \frac{\partial u}{\partial x_{p2}} \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      \frac{\partial u}{\partial x_{1q}} & \frac{\partial u}{\partial x_{2q}} & \cdots & \frac{\partial u}{\partial x_{pq}} 
      \end{bmatrix}.
      $$

      
      Combining the terms, we get:

     
      $$
      \boxed{\frac{\partial (uv)}{\partial \mathbf{X}} = u \frac{\partial v}{\partial \mathbf{X}} + v \frac{\partial u}{\partial \mathbf{X}}.}
      $$

    </div>
  </details>
</div>


<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial (\mathbf{a}^\top \mathbf{X} \mathbf{b})}{\partial \mathbf{X}} = \mathbf{b} \mathbf{a}^\top$$
    </summary>
    <div style="margin-top: 10px;">

      
      Let \( \mathbf{a} \in \mathbb{R}^m \) and \( \mathbf{b} \in \mathbb{R}^n \) be constant vectors, and let \( \mathbf{X} \in \mathbb{R}^{m \times n} \) be a matrix.  
      The scalar \( \mathbf{a}^\top \mathbf{X} \mathbf{b} \) is defined as:

      $$
      \mathbf{a}^\top \mathbf{X} \mathbf{b} = \sum_{i=1}^m \sum_{j=1}^n a_i X_{ij} b_j
      $$

      We aim to compute the derivative of \( \mathbf{a}^\top \mathbf{X} \mathbf{b} \) with respect to \( \mathbf{X} \).

      
      As we know that The derivative of a scalar function \( y \) with respect to a matrix \( \mathbf{X} \) is given by:

      $$
      \frac{\partial y}{\partial \mathbf{X}} = 
      \begin{bmatrix} 
      \frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{21}} & \cdots & \frac{\partial y}{\partial x_{p1}} \\ 
      \frac{\partial y}{\partial x_{12}} & \frac{\partial y}{\partial x_{22}} & \cdots & \frac{\partial y}{\partial x_{p2}} \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      \frac{\partial y}{\partial x_{1q}} & \frac{\partial y}{\partial x_{2q}} & \cdots & \frac{\partial y}{\partial x_{pq}} 
      \end{bmatrix}.
      $$

      
      The scalar \( \mathbf{a}^\top \mathbf{X} \mathbf{b} \) can be expanded as:

      $$
      \mathbf{a}^\top \mathbf{X} \mathbf{b} = \sum_{i=1}^m \sum_{j=1}^n a_i X_{ij} b_j.
      $$

      Taking the derivative with respect to \( X_{ij} \), we get:

      $$
      \frac{\partial (\mathbf{a}^\top \mathbf{X} \mathbf{b})}{\partial X_{ij}} = a_i b_j.
      $$

      
      From above This can be written as:

      $$
      \frac{\partial (\mathbf{a}^\top \mathbf{X} \mathbf{b})}{\partial \mathbf{X}} = 
      \begin{bmatrix} 
      a_1 b_1 & a_2 b_1 & \cdots & a_m b_1 \\ 
      a_1 b_2 & a_2 b_2 & \cdots & a_m b_2 \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      a_1 b_n & a_2 b_n & \cdots & a_m b_n 
      \end{bmatrix}.
      $$

      
      The matrix above is the outer product of \( \mathbf{b} \) and \( \mathbf{a} \). The outer product of two vectors \( \mathbf{b} \in \mathbb{R}^n \) and \( \mathbf{a} \in \mathbb{R}^m \) is defined as:

      $$
      \mathbf{b} \mathbf{a}^\top = 
      \begin{bmatrix} 
      b_1 \\ 
      b_2 \\ 
      \vdots \\ 
      b_n 
      \end{bmatrix}
      \begin{bmatrix} 
      a_1 & a_2 & \cdots & a_m 
      \end{bmatrix}
      =
      \begin{bmatrix} 
      b_1 a_1 & b_1 a_2 & \cdots & b_1 a_m \\ 
      b_2 a_1 & b_2 a_2 & \cdots & b_2 a_m \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      b_n a_1 & b_n a_2 & \cdots & b_n a_m 
      \end{bmatrix}.
      $$

      Thus, we conclude:

      $$
      \frac{\partial (\mathbf{a}^\top \mathbf{X} \mathbf{b})}{\partial \mathbf{X}} = \mathbf{b} \mathbf{a}^\top.
      $$

      
      The derivative of \( \mathbf{a}^\top \mathbf{X} \mathbf{b} \) with respect to \( \mathbf{X} \) is:

      $$
      \boxed{\frac{\partial (\mathbf{a}^\top \mathbf{X} \mathbf{b})}{\partial \mathbf{X}} = \mathbf{b} \mathbf{a}^\top.}
      $$

    </div>
  </details>
</div>


<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial \operatorname{tr}(\mathbf{X})}{\partial \mathbf{X}} = \mathbf{I}$$
    </summary>
    <div style="margin-top: 10px;">

      
      Let \( \mathbf{X} \in \mathbb{R}^{n \times n} \) be a square matrix.  
      The trace of \( \mathbf{X} \), denoted as \( \operatorname{tr}(\mathbf{X}) \), is defined as the sum of its diagonal elements:

      $$
      \operatorname{tr}(\mathbf{X}) = \sum_{i=1}^n X_{ii}.
      $$

     

      
      

      Taking the derivative with respect to \( X_{ij} \), we get:

      $$
      \frac{\partial \operatorname{tr}(\mathbf{X})}{\partial X_{ij}} = 
      \begin{cases} 
      1 & \text{if } i = j, \\ 
      0 & \text{if } i \neq j.
      \end{cases}
      $$

       The derivative \( \frac{\partial \operatorname{tr}(\mathbf{X})}{\partial \mathbf{X}} \) is a matrix where the diagonal entries are \( 1 \) (since \( i = j \)) and the off-diagonal entries are \( 0 \) (since \( i \neq j \)). This is the identity matrix:

      $$
      \frac{\partial \operatorname{tr}(\mathbf{X})}{\partial \mathbf{X}} = 
      \begin{bmatrix} 
      1 & 0 & \cdots & 0 \\ 
      0 & 1 & \cdots & 0 \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      0 & 0 & \cdots & 1 
      \end{bmatrix} = \mathbf{I}.
      $$

      
      The derivative of \( \operatorname{tr}(\mathbf{X}) \) with respect to \( \mathbf{X} \) is:

      $$
      \boxed{\frac{\partial \operatorname{tr}(\mathbf{X})}{\partial \mathbf{X}} = \mathbf{I}.}
      $$

    </div>
  </details>
</div>


<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial \operatorname{tr}(\mathbf{AX})}{\partial \mathbf{X}} = \frac{\partial \operatorname{tr}(\mathbf{XA})}{\partial \mathbf{X}} = \mathbf{A}$$
    </summary>
    <div style="margin-top: 10px;">

      
      Let \( \mathbf{A} \in \mathbb{R}^{n \times n} \) be a constant matrix, and \( \mathbf{X} \in \mathbb{R}^{n \times n} \) be a variable matrix.  
      The trace of \( \mathbf{AX} \) or \( \mathbf{XA} \), denoted as \( \operatorname{tr}(\mathbf{AX}) \) or \( \operatorname{tr}(\mathbf{XA}) \), is defined as:

      $$
      \operatorname{tr}(\mathbf{AX}) = \sum_{i=1}^n \sum_{j=1}^n A_{ij} X_{ji}.
      $$

      We aim to compute the derivative of \( \operatorname{tr}(\mathbf{AX}) \) and \( \operatorname{tr}(\mathbf{XA}) \) with respect to \( \mathbf{X} \).

      
      As we know The derivative of a scalar function \( y \) with respect to a matrix \( \mathbf{X} \) is given by:

      $$
      \frac{\partial y}{\partial \mathbf{X}} = 
      \begin{bmatrix} 
      \frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{21}} & \cdots & \frac{\partial y}{\partial x_{n1}} \\ 
      \frac{\partial y}{\partial x_{12}} & \frac{\partial y}{\partial x_{22}} & \cdots & \frac{\partial y}{\partial x_{n2}} \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      \frac{\partial y}{\partial x_{1n}} & \frac{\partial y}{\partial x_{2n}} & \cdots & \frac{\partial y}{\partial x_{nn}} 
      \end{bmatrix}.
      $$

     
      The trace \( \operatorname{tr}(\mathbf{AX}) \) can be expanded as:

      $$
      \operatorname{tr}(\mathbf{AX}) = \sum_{i=1}^n \sum_{j=1}^n A_{ij} X_{ji}.
      $$

      Taking the derivative with respect to \( X_{kl} \), we get:

      $$
      \frac{\partial \operatorname{tr}(\mathbf{AX})}{\partial X_{kl}} = A_{lk}.
      $$

      Similarly, for \( \operatorname{tr}(\mathbf{XA}) \), we can write:

      $$
      \operatorname{tr}(\mathbf{XA}) = \sum_{i=1}^n \sum_{j=1}^n X_{ij} A_{ji}.
      $$

      Taking the derivative with respect to \( X_{kl} \), we get:

      $$
      \frac{\partial \operatorname{tr}(\mathbf{XA})}{\partial X_{kl}} = A_{lk}.
      $$

      
      Using the identity, the derivative \( \frac{\partial \operatorname{tr}(\mathbf{AX})}{\partial \mathbf{X}} \) is a matrix where the \((k, l)\)-th entry is \( A_{lk} \). This is simply the matrix \( \mathbf{A} \):

      $$
      \frac{\partial \operatorname{tr}(\mathbf{AX})}{\partial \mathbf{X}} = \mathbf{A}.
      $$

      Similarly, for \( \operatorname{tr}(\mathbf{XA}) \), the derivative is also:

      $$
      \frac{\partial \operatorname{tr}(\mathbf{XA})}{\partial \mathbf{X}} = \mathbf{A}.
      $$

      
      The derivatives of \( \operatorname{tr}(\mathbf{AX}) \) and \( \operatorname{tr}(\mathbf{XA}) \) with respect to \( \mathbf{X} \) are:

      $$
      \boxed{\frac{\partial \operatorname{tr}(\mathbf{AX})}{\partial \mathbf{X}} = \frac{\partial \operatorname{tr}(\mathbf{XA})}{\partial \mathbf{X}} = \mathbf{A}.}
      $$

    </div>
  </details>
</div>

<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X})}{\partial \mathbf{X}} = \mathbf{X}^\top (\mathbf{A} + \mathbf{A}^\top)$$
    </summary>
    <div style="margin-top: 10px;">

      
      Let \( \mathbf{A} \in \mathbb{R}^{n \times n} \) be a constant matrix (not a function of \( \mathbf{X} \)), and \( \mathbf{X} \in \mathbb{R}^{n \times n} \) be a variable matrix.  

      The trace of \( \mathbf{X}^\top \mathbf{A} \mathbf{X} \), denoted as \( \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X}) \), is defined as:

      $$
      \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X}) = \sum_{i=1}^n \sum_{j=1}^n \sum_{k=1}^n X_{ki} A_{kj} X_{ji}.
      $$

      We aim to compute the derivative of \( \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X}) \) with respect to \( \mathbf{X} \) using the numerator layout convention, where the derivative matrix’s \( (p, q) \)-th entry is the partial derivative with respect to \( X_{pq} \).

      
      In numerator layout, the derivative of a scalar function \( y \) with respect to a matrix \( \mathbf{X} \) is a matrix of the same size as \( \mathbf{X} \), defined as:

      $$
      \frac{\partial y}{\partial \mathbf{X}} = 
      \begin{bmatrix} 
      \frac{\partial y}{\partial X_{11}} & \frac{\partial y}{\partial X_{21}} & \cdots & \frac{\partial y}{\partial X_{n1}} \\ 
      \frac{\partial y}{\partial X_{12}} & \frac{\partial y}{\partial X_{22}} & \cdots & \frac{\partial y}{\partial X_{n2}} \\ 
      \vdots & \vdots & \ddots & \vdots \\ 
      \frac{\partial y}{\partial X_{1n}} & \frac{\partial y}{\partial X_{2n}} & \cdots & \frac{\partial y}{\partial X_{nn}} 
      \end{bmatrix}.
      $$

      
      The trace is:

      $$
      \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X}) = \sum_{i=1}^n \sum_{j=1}^n \sum_{k=1}^n X_{ki} A_{kj} X_{ji}.
      $$

      Compute the partial derivative with respect to \( X_{pq} \). Since \( X_{pq} \) appears in two positions (\( X_{ki} \) and \( X_{ji} \)), 
      $$  $$
      we calculate both contributions:
      $$  $$
      a) Contribution from \( X_{ki} \): Set \( k = p \), \( i = q \):
        $$
        \frac{\partial}{\partial X_{pq}} \sum_{i=1}^n \sum_{k=1}^n \sum_{j=1}^n X_{ki} A_{kj} X_{ji} = \sum_{j=1}^n A_{pj} X_{jq}.
        $$
        (When \( k = p \), \( i = q \), the term \( X_{pq} A_{pj} X_{jq} \) differentiates to \( A_{pj} X_{jq} \).)
         
         $$ $$
      b)Contribution from \( X_{ji} \): Set \( j = p \), \( i = q \):
        $$
        \frac{\partial}{\partial X_{pq}} \sum_{i=1}^n \sum_{k=1}^n \sum_{j=1}^n X_{ki} A_{kj} X_{ji} = \sum_{k=1}^n X_{kq} A_{kp}.
        $$
        (When \( j = p \), \( i = q \), the term \( X_{kq} A_{kp} X_{qp} \) differentiates to \( X_{kq} A_{kp} \).)

      Total partial derivative:
      $$
      \frac{\partial \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X})}{\partial X_{pq}} = \sum_{j=1}^n A_{pj} X_{jq} + \sum_{k=1}^n X_{kq} A_{kp}.
      $$

      
      The derivative matrix has entries:
      $$
      \left( \frac{\partial \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X})}{\partial \mathbf{X}} \right)_{pq} = \sum_{j=1}^n A_{pj} X_{jq} + \sum_{k=1}^n X_{kq} A_{kp}.
      $$
      Rewrite using index substitution (let \( m \) replace \( j \) and \( k \)):
      $$ \sum_{j=1}^n A_{pj} X_{jq} = \sum_{m=1}^n A_{pm} X_{mq} $$,
      $$ \sum_{k=1}^n X_{kq} A_{kp} = \sum_{m=1}^n X_{mq} A_{mp} $$.

      

      Combine:
      $$
      \frac{\partial \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X})}{\partial X_{pq}} = \sum_{m=1}^n X_{mp} A_{mq} + \sum_{m=1}^n X_{mp} A_{qm} = \sum_{m=1}^n X_{mp} (A_{mq} + A_{qm}).
      $$
      This is the \( (p, q) \)-th entry of \( \mathbf{X}^\top (\mathbf{A} + \mathbf{A}^\top) \), since:
      $$
      [\mathbf{X}^\top (\mathbf{A} + \mathbf{A}^\top)]_{pq} = \sum_{m=1}^n X_{mp} (A_{mq} + A_{qm}).
      $$

      Thus:
      $$
      \frac{\partial \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X})}{\partial \mathbf{X}} = \mathbf{X}^\top (\mathbf{A} + \mathbf{A}^\top).
      $$

      
      
      The derivative of \( \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X}) \) with respect to \( \mathbf{X} \) in numerator layout is:

      $$
      \boxed{\frac{\partial \operatorname{tr}(\mathbf{X}^\top \mathbf{A} \mathbf{X})}{\partial \mathbf{X}} = \mathbf{X}^\top (\mathbf{A} + \mathbf{A}^\top)}
      $$

    </div>
  </details>
</div>



<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\operatorname{tr}(\mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N) = \operatorname{tr}(\mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1}) = \cdots$$
    </summary>
    <div style="margin-top: 10px;">

      
      Let \( \mathbf{A}_1, \mathbf{A}_2, \dots, \mathbf{A}_N \) be matrices of compatible dimensions such that their product \( \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N \) is square.  
      The trace of the product \( \operatorname{tr}(\mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N) \) satisfies the **cyclic property**:

      $$
      \operatorname{tr}(\mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N) = \operatorname{tr}(\mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1}) = \cdots.
      $$

      
      The trace of a square matrix \( \mathbf{M} \in \mathbb{R}^{n \times n} \) is defined as the sum of its diagonal elements:

      $$
      \operatorname{tr}(\mathbf{M}) = \sum_{i=1}^n M_{ii}.
      $$

      For the product \( \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N \), the \((i, i)\)-th diagonal entry is:

      $$
      (\mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N)_{ii} = \sum_{j_1, j_2, \dots, j_{N-1}} (\mathbf{A}_1)_{ij_1} (\mathbf{A}_2)_{j_1 j_2} \cdots (\mathbf{A}_N)_{j_{N-1} i}.
      $$

      Therefore, the trace of \( \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N \) is:

      $$
      \operatorname{tr}(\mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N) = \sum_{i=1}^n \sum_{j_1, j_2, \dots, j_{N-1}} (\mathbf{A}_1)_{ij_1} (\mathbf{A}_2)_{j_1 j_2} \cdots (\mathbf{A}_N)_{j_{N-1} i}.
      $$

      
      Now consider a cyclic permutation of the matrices, such as \( \operatorname{tr}(\mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1}) \). The \((i, i)\)-th diagonal entry of \( \mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1} \) is:

      $$
      (\mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1})_{ii} = \sum_{j_1, j_2, \dots, j_{N-1}} (\mathbf{A}_N)_{ij_{N-1}} (\mathbf{A}_1)_{j_{N-1} j_1} (\mathbf{A}_2)_{j_1 j_2} \cdots (\mathbf{A}_{N-1})_{j_{N-2} i}.
      $$

      The trace of \( \mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1} \) is:

      $$
      \operatorname{tr}(\mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1}) = \sum_{i=1}^n \sum_{j_1, j_2, \dots, j_{N-1}} (\mathbf{A}_N)_{ij_{N-1}} (\mathbf{A}_1)_{j_{N-1} j_1} (\mathbf{A}_2)_{j_1 j_2} \cdots (\mathbf{A}_{N-1})_{j_{N-2} i}.
      $$

      By reordering the summation indices \( i, j_1, j_2, \dots, j_{N-1} \), we see that:

      $$
      \operatorname{tr}(\mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1}) = \operatorname{tr}(\mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N).
      $$

      
      This argument can be repeated for any cyclic permutation of the matrices \( \mathbf{A}_1, \mathbf{A}_2, \dots, \mathbf{A}_N \). Thus, the trace is invariant under cyclic permutations:

      $$
      \operatorname{tr}(\mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N) = \operatorname{tr}(\mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1}) = \cdots.
      $$

      
      The trace of a product of matrices is invariant under cyclic permutations:

      $$
      \boxed{\operatorname{tr}(\mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_N) = \operatorname{tr}(\mathbf{A}_N \mathbf{A}_1 \mathbf{A}_2 \cdots \mathbf{A}_{N-1}) = \cdots.}
      $$

    </div>
  </details>
</div>


<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$\frac{\partial \operatorname{tr}(\mathbf{BAX})}{\partial \mathbf{X}} = \mathbf{BA}$$
    </summary>
    <div style="margin-top: 10px;">

      
    </div>
  </details>
</div>


 You can refer to this [ wikipedia article](https://en.wikipedia.org/wiki/Matrix_calculus) for more details and identities.
