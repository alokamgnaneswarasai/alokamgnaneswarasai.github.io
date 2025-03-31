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


### Example: Partial Differentiation of $f(x, y) = x^2 + y^2$

- **Problem Statement**: Compute the partial derivatives of $f(x, y) = x^2 + y^2$ with respect to $x$ and $y$, and interpret their meaning at the point $(1, 2)$.
- **Mathematical Solution**:
  1. **Partial Derivative with Respect to $x$**:
     - Treat $y$ as a constant:  
       $$ \frac{\partial f}{\partial x} = \frac{\partial}{\partial x} (x^2 + y^2) = 2x + 0 = 2x $$
     - At $(x, y) = (1, 2)$:  
       $$ \frac{\partial f}{\partial x} \bigg|_{(1, 2)} = 2 \cdot 1 = 2 $$
  2. **Partial Derivative with Respect to $y$**:
     - Treat $x$ as a constant:  
       $$ \frac{\partial f}{\partial y} = \frac{\partial}{\partial y} (x^2 + y^2) = 0 + 2y = 2y $$
     - At $(x, y) = (1, 2)$:  
       $$ \frac{\partial f}{\partial y} \bigg|_{(1, 2)} = 2 \cdot 2 = 4 $$
- **Interpretation**:
  - $\frac{\partial f}{\partial x} = 2$ at $(1, 2)$: The rate of change of $f$ in the $x$-direction is 2 units per unit increase in $x$, with $y$ fixed at 2.
  - $\frac{\partial f}{\partial y} = 4$ at $(1, 2)$: The rate of change of $f$ in the $y$-direction is 4 units per unit increase in $y$, with $x$ fixed at 1.
- These partial derivatives describe the slopes of the tangent lines to the surface $f(x, y) = x^2 + y^2$ along the $x$ and $y$ axes at the point $(1, 2)$.


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

-------