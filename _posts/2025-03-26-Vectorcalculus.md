---
layout: post
title: Introduction to Univariate calculus
tags: [calculus,differentiation,minima,maxima,inflection]
author: Gnaneswara Sai
---

# The Role of Calculus in Machine and Deep Learning

## Why Calculus Matters in Machine and Deep Learning

Calculus is the backbone of machine learning (ML) and deep learning (DL). At its core, these fields aim to optimize models—think neural networks or regression models—by minimizing error or loss functions. Calculus provides the tools to do this efficiently. Differentiation, in particular, is crucial because it tells us how a function changes with respect to its inputs, enabling algorithms like gradient descent to "learn" by adjusting parameters in the direction that reduces error. Without differentiation, we’d be stuck guessing parameter updates blindly, which is impractical for complex models with millions of variables.

In ML/DL, we need differentiation to:
- Compute gradients (rates of change) of loss functions with respect to weights and biases.
- Optimize models iteratively by following these gradients.
- Understand how small changes in input affect predictions, which ties into stability and generalization.

Simply put, calculus turns the abstract goal of "learning" into a concrete, computable process.

## What is Differentiation?

Differentiation measures how much a function’s output changes when its input changes. Imagine a function $ f(x) $ as a machine: you tweak the input $ x $ by a tiny amount, and differentiation tells you how much the output $ f(x) $ shifts. Mathematically, the derivative of $ f(x) $ at a point $ x $, denoted $ f'(x) $ or $ \frac{df}{dx} $, is the slope of the tangent line to the function’s graph at that point.

### The First Principle of Differentiation

The formal definition of the derivative comes from the *first principle*. It’s the limit of the average rate of change over an infinitesimally small interval:

$$ 
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

Here, $ h $ is a tiny change in $ x $, and the expression inside the limit is the difference quotient—the slope of a secant line that becomes a tangent as $ h $ shrinks to zero. Let’s apply this to compute some derivatives.

#### Example 1: Derivative of $ f(x) = x^2 $
Using the first principle:
$$
\begin{align*}
f'(x) & = \lim_{h \to 0} \frac{(x + h)^2 - x^2}{h} \\
      & = \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} \\
      & = \lim_{h \to 0} \frac{2xh + h^2}{h} \\
      & = \lim_{h \to 0} (2x + h) \\
      & = 2x
\end{align*}
$$

#### Example 2: Derivative of $ f(x) = 3x + 1 $
$$
\begin{align*}
f'(x) & = \lim_{h \to 0} \frac{[3(x + h) + 1] - [3x + 1]}{h} \\
      & = \lim_{h \to 0} \frac{3x + 3h + 1 - 3x - 1}{h} \\
      & = \lim_{h \to 0} \frac{3h}{h} \\
      & = 3
\end{align*}
$$

#### Example 3: Derivative of $ f(x) = \frac{1}{x} $ (for $ x \neq 0 $)
$$
\begin{align*}
f'(x) & = \lim_{h \to 0} \frac{\frac{1}{x + h} - \frac{1}{x}}{h} \\
      & = \lim_{h \to 0} \frac{\frac{x - (x + h)}{x(x + h)}}{h} \\
      & = \lim_{h \to 0} \frac{\frac{-h}{x(x + h)}}{h} \\
      & = \lim_{h \to 0} \frac{-1}{x(x + h)} \\
      & = \frac{-1}{x^2}
\end{align*}
$$

These examples show how the first principle systematically computes the rate of change.

## Continuity and Differentiability

### Continuity
A function $ f(x) $ is continuous at a point $ x = a $ if:
1. $ f(a) $ is defined.
2. The limit $ \lim_{x \to a} f(x) $ exists.
3. $ \lim_{x \to a} f(x) = f(a) $.

In plain terms, the graph has no breaks, jumps, or holes at $ x = a $. Continuity ensures the function behaves predictably as $ x $ approaches $ a $.

**Example**: $ f(x) = \lvert x \rvert $ is continuous at $ x = 0 $ because:
- $ f(0) = 0 $,
- Left limit: $ \lim_{x \to 0^-} \lvert x \rvert = 0 $,
- Right limit: $ \lim_{x \to 0^+} \lvert x \rvert = 0 $,
- Both equal $ f(0) $.

### Differentiability
A function is differentiable at \( x = a \) if its derivative exists there—i.e., the limit from the first principle is finite and well-defined. Differentiability implies continuity (but not vice versa). Geometrically, it means the function has a unique tangent line at that point, with no sharp corners or vertical slopes.

**Example**: $ f(x) = \lvert x \rvert $ is not differentiable at $ x = 0 $. Check the left and right derivatives:
- Left: 
  $$
  \lim_{h \to 0^-} \frac{\lvert 0 + h \rvert - \lvert 0 \rvert}{h} = \lim_{h \to 0^-} \frac{\lvert h \rvert}{h} = \lim_{h \to 0^-} \frac{-h}{h} = -1
  $$
- Right: 
  $$
  \lim_{h \to 0^+} \frac{\lvert 0 + h \rvert - \lvert 0 \rvert}{h} = \lim_{h \to 0^+} \frac{h}{h} = 1
  $$

Since $ -1 \neq 1 $, the derivative doesn’t exist at \( x = 0 \), despite continuity.

## Rules for Basic Functions

Here are the differentiation rules for some basic functions:

### 1. Power Rule
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( x^a \right) = a x^{a-1} $$
    </summary>
    <div style="margin-top: 5px;">
      Using the first principle:
      $$
      \frac{d}{dx} \left( x^a \right) = \lim_{h \to 0} \frac{(x + h)^a - x^a}{h}
      $$
      Expand \( (x + h)^a \) using the Binomial Theorem:
      $$
      = \lim_{h \to 0} \frac{x^a + a x^{a-1} h + \frac{a(a-1)}{2} x^{a-2} h^2 + \dots - x^a}{h}
      $$
      Simplify:
      $$
      = \lim_{h \to 0} \left( a x^{a-1} + \frac{a(a-1)}{2} x^{a-2} h + \dots \right)
      $$
      As \( h \to 0 \), all terms with \( h \) vanish:
      $$
      \frac{d}{dx} \left( x^a \right) = a x^{a-1}
      $$
    </div>
  </details>
</div>

### 2. Trigonometric Functions

#### a. Sine Function
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( \sin(x) \right) = \cos(x) $$
    </summary>
    <div style="margin-top: 5px;">
      Using the first principle:
      $$
      \frac{d}{dx} \left( \sin(x) \right) = \lim_{h \to 0} \frac{\sin(x + h) - \sin(x)}{h}
      $$
      Using the trigonometric identity:
      $$
      \sin(x + h) = \sin(x) \cos(h) + \cos(x) \sin(h)
      $$
      Substitute:
      $$
      = \lim_{h \to 0} \frac{\sin(x) \cos(h) + \cos(x) \sin(h) - \sin(x)}{h}
      $$
      Factor terms:
      $$
      = \lim_{h \to 0} \left[ \sin(x) \frac{\cos(h) - 1}{h} + \cos(x) \frac{\sin(h)}{h} \right]
      $$
      Using \( \lim_{h \to 0} \frac{\sin(h)}{h} = 1 \) and \( \lim_{h \to 0} \frac{\cos(h) - 1}{h} = 0 \):
      $$
      = \cos(x)
      $$
    </div>
  </details>
</div>

#### b. Cosine Function
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( \cos(x) \right) = -\sin(x) $$
    </summary>
    <div style="margin-top: 5px;">
      Using the first principle:
      $$
      \frac{d}{dx} \left( \cos(x) \right) = \lim_{h \to 0} \frac{\cos(x + h) - \cos(x)}{h}
      $$
      Using the trigonometric identity:
      $$
      \cos(x + h) = \cos(x) \cos(h) - \sin(x) \sin(h)
      $$
      Substitute:
      $$
      = \lim_{h \to 0} \frac{\cos(x) \cos(h) - \sin(x) \sin(h) - \cos(x)}{h}
      $$
      Factor terms:
      $$
      = \lim_{h \to 0} \left[ \cos(x) \frac{\cos(h) - 1}{h} - \sin(x) \frac{\sin(h)}{h} \right]
      $$
      Using \( \lim_{h \to 0} \frac{\sin(h)}{h} = 1 \) and \( \lim_{h \to 0} \frac{\cos(h) - 1}{h} = 0 \):
      $$
      = -\sin(x)
      $$
    </div>
  </details>
</div>

#### c. Tangent Function
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( \tan(x) \right) = \sec^2(x) $$
    </summary>
    <div style="margin-top: 5px;">
      Using the identity \( \tan(x) = \frac{\sin(x)}{\cos(x)} \), apply the quotient rule:
      $$
      \frac{d}{dx} \left( \tan(x) \right) = \frac{\cos(x) \cdot \frac{d}{dx} \sin(x) - \sin(x) \cdot \frac{d}{dx} \cos(x)}{\cos^2(x)}
      $$
      Substitute \( \frac{d}{dx} \sin(x) = \cos(x) \) and \( \frac{d}{dx} \cos(x) = -\sin(x) \):
      $$
      = \frac{\cos(x) \cdot \cos(x) - \sin(x) \cdot (-\sin(x))}{\cos^2(x)}
      $$
      Simplify:
      $$
      = \frac{\cos^2(x) + \sin^2(x)}{\cos^2(x)}
      $$
      Using the Pythagorean identity \( \sin^2(x) + \cos^2(x) = 1 \):
      $$
      = \frac{1}{\cos^2(x)} = \sec^2(x)
      $$
    </div>
  </details>
</div>

### 3. Exponential Function
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( e^x \right) = e^x $$
    </summary>
    <div style="margin-top: 5px;">
      Using the first principle:
      $$
      \frac{d}{dx} \left( e^x \right) = \lim_{h \to 0} \frac{e^{x+h} - e^x}{h}
      $$
      Factor \( e^x \):
      $$
      = \lim_{h \to 0} \frac{e^x \left( e^h - 1 \right)}{h}
      $$
      Since \( \lim_{h \to 0} \frac{e^h - 1}{h} = 1 \):
      $$
      \frac{d}{dx} \left( e^x \right) = e^x
      $$
    </div>
  </details>
</div>

### 4. Logarithmic Function
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( \ln(x) \right) = \frac{1}{x} $$
    </summary>
    <div style="margin-top: 5px;">
      Using the first principle:
      $$
      \frac{d}{dx} \left( \ln(x) \right) = \lim_{h \to 0} \frac{\ln(x + h) - \ln(x)}{h}
      $$
      Using the logarithmic property \( \ln(a) - \ln(b) = \ln\left(\frac{a}{b}\right) \):
      $$
      = \lim_{h \to 0} \frac{\ln\left(\frac{x + h}{x}\right)}{h}
      $$
      Simplify:
      $$
      = \lim_{h \to 0} \frac{\ln\left(1 + \frac{h}{x}\right)}{h}
      $$
      Using the approximation \( \ln(1 + u) \approx u \) for small \( u \):
      $$
      = \lim_{h \to 0} \frac{\frac{h}{x}}{h} = \frac{1}{x}
      $$
    </div>
  </details>
</div>

### 5. Inverse Trigonometric Functions

#### a. Arcsine Function
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 1px; padding: 1px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 5px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( \arcsin(x) \right) = \frac{1}{\sqrt{1 - x^2}} $$
    </summary>
    <div style="margin-top: 1px;">
      Using the identity \( \sin(y) = x \), we have:
      $$
      \frac{d}{dx} \arcsin(x) = \frac{1}{\cos(y)}
      $$
      From the Pythagorean identity \( \cos^2(y) = 1 - \sin^2(y) \):
      $$
      \cos(y) = \sqrt{1 - x^2}
      $$
      Thus:
      $$
      \frac{d}{dx} \arcsin(x) = \frac{1}{\sqrt{1 - x^2}}
      $$
    </div>
  </details>
</div>

#### b. Arccosine Function
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( \arccos(x) \right) = \frac{-1}{\sqrt{1 - x^2}} $$
    </summary>
    <div style="margin-top: 5px;">
      Using the identity \( \cos(y) = x \), we have:
      $$
      \frac{d}{dx} \arccos(x) = -\frac{1}{\sin(y)}
      $$
      From the Pythagorean identity \( \sin^2(y) = 1 - \cos^2(y) \):
      $$
      \sin(y) = \sqrt{1 - x^2}
      $$
      Thus:
      $$
      \frac{d}{dx} \arccos(x) = \frac{-1}{\sqrt{1 - x^2}}
      $$
    </div>
  </details>
</div>

#### c. Arctangent Function
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 5px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( \arctan(x) \right) = \frac{1}{1 + x^2} $$
    </summary>
    <div style="margin-top: 5px;">
      Using the identity \( \tan(y) = x \), we have:
      $$
      \frac{d}{dx} \arctan(x) = \frac{1}{\sec^2(y)}
      $$
      From the trigonometric identity \( \sec^2(y) = 1 + \tan^2(y) \):
      $$
      \sec^2(y) = 1 + x^2
      $$
      Thus:
      $$
      \frac{d}{dx} \arctan(x) = \frac{1}{1 + x^2}
      $$
    </div>
  </details>
</div>

---

## Rules for Combined Functions

### 1. Constant Rule
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 1px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( c \right) = 0 $$
    </summary>
    <div style="margin-top: 0px;">     
      The derivative of a constant is always zero because a constant does not change with respect to \( x \).
    </div>
  </details>
</div>

### 2. Sum Rule
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; background-color: #f9f9f9; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); margin-bottom: 1px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( f(x) + g(x) \right) = f'(x) + g'(x) $$
    </summary>
    <div style="margin-top: 0px;">
      Using the first principle:
      $$
      \frac{d}{dx} \left( f(x) + g(x) \right) = \lim_{h \to 0} \frac{[f(x + h) + g(x + h)] - [f(x) + g(x)]}{h}
      $$
      Simplify:
      $$
      = \lim_{h \to 0} \left[ \frac{f(x + h) - f(x)}{h} + \frac{g(x + h) - g(x)}{h} \right]
      $$
      As \( h \to 0 \):
      $$
      = f'(x) + g'(x)
      $$
    </div>
  </details>
</div>

### 3. Product Rule
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); margin-bottom: 15px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( f(x) g(x) \right) = f'(x) g(x) + f(x) g'(x) $$
    </summary>
    <div style="margin-top: 10px;">
      Using the first principle:
      $$
      \frac{d}{dx} \left( f(x) g(x) \right) = \lim_{h \to 0} \frac{f(x + h) g(x + h) - f(x) g(x)}{h}
      $$
      Add and subtract \( f(x) g(x + h) \):
      $$
      = \lim_{h \to 0} \frac{f(x + h) g(x + h) - f(x) g(x + h) + f(x) g(x + h) - f(x) g(x)}{h}
      $$
      Factor terms:
      $$
      = \lim_{h \to 0} \left[ g(x + h) \frac{f(x + h) - f(x)}{h} + f(x) \frac{g(x + h) - g(x)}{h} \right]
      $$
      As \( h \to 0 \):
      $$
      \frac{d}{dx} \left( f(x) g(x) \right) = f'(x) g(x) + f(x) g'(x)
      $$
    </div>
  </details>
</div>

### 4. Quotient Rule
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); margin-bottom: 15px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = \frac{f'(x) g(x) - f(x) g'(x)}{g(x)^2} $$
    </summary>
    <div style="margin-top: 10px;">
      To derive the quotient rule, we first compute the derivative of \( \frac{1}{g(x)} \) and then use the product rule.

      #### Step 1: Derivative of \( \frac{1}{g(x)} \)
      Let \( h(x) = \frac{1}{g(x)} \). Using the chain rule:
      $$
      h'(x) = \frac{d}{dx} \left( g(x)^{-1} \right) = -g(x)^{-2} \cdot g'(x)
      $$
      Thus:
      $$
      \frac{d}{dx} \left( \frac{1}{g(x)} \right) = -\frac{g'(x)}{g(x)^2}
      $$

      #### Step 2: Use the Product Rule
      Let \( q(x) = \frac{f(x)}{g(x)} = f(x) \cdot \frac{1}{g(x)} \). Using the product rule:
      $$
      \frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = \frac{d}{dx} \left( f(x) \cdot \frac{1}{g(x)} \right)
      $$
      Apply the product rule:
      $$
      \frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = f'(x) \cdot \frac{1}{g(x)} + f(x) \cdot \frac{d}{dx} \left( \frac{1}{g(x)} \right)
      $$
      Substitute \( \frac{d}{dx} \left( \frac{1}{g(x)} \right) = -\frac{g'(x)}{g(x)^2} \):
      $$
      \frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = \frac{f'(x)}{g(x)} - f(x) \cdot \frac{g'(x)}{g(x)^2}
      $$
      Simplify:
      $$
      \frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = \frac{f'(x) g(x) - f(x) g'(x)}{g(x)^2}
      $$

      Thus, the quotient rule is derived:
      $$
      \boxed{\frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = \frac{f'(x) g(x) - f(x) g'(x)}{g(x)^2}}
      $$
    </div>
  </details>
</div>

### 5. Chain Rule
<div class="equation-box" style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); margin-bottom: 15px;">
  <details>
    <summary style="cursor: pointer; font-weight: bold; color: #333;">
      $$ \frac{d}{dx} \left( f(g(x)) \right) = f'(g(x)) \cdot g'(x) $$
    </summary>
    <div style="margin-top: 10px;">
      Using the first principle:
      $$
      \frac{d}{dx} \left( f(g(x)) \right) = \lim_{h \to 0} \frac{f(g(x + h)) - f(g(x))}{h}
      $$
      Let \( u = g(x + h) - g(x) \), so \( h = \frac{u}{g'(x)} \):
      $$
      = \lim_{u \to 0} \frac{f(g(x) + u) - f(g(x))}{u} \cdot g'(x)
      $$
      By the definition of \( f'(g(x)) \):
      $$
      \frac{d}{dx} \left( f(g(x)) \right) = f'(g(x)) \cdot g'(x)
      $$
    </div>
  </details>
</div>

---

## Local Minima and Local Maxima

### Local Minima
A function $f(x)$ has a **local minimum** at $x = c$ if:
1. $f(c)$ is defined.
2. There exists an interval $(a, b)$ containing $c$ such that $f(c) \leq f(x)$ for all $x \in (a, b)$.

In simpler terms, $f(c)$ is smaller than or equal to the values of $f(x)$ in a small neighborhood around $c$.

### Local Maxima
A function $f(x)$ has a **local maximum** at $x = c$ if:
1. $f(c)$ is defined.
2. There exists an interval $(a, b)$ containing $c$ such that $f(c) \geq f(x)$ for all $x \in (a, b)$.

In simpler terms, $f(c)$ is greater than or equal to the values of $f(x)$ in a small neighborhood around $c$.

---

## First Derivative Test

The **first derivative test** helps determine whether a critical point is a local minimum, local maximum, or neither.

### Steps:
1. Find the critical points of $f(x)$ by solving $f'(x) = 0$ or where $f'(x)$ is undefined.
2. Analyze the sign of $f'(x)$ on intervals around each critical point:
   - If $f'(x)$ changes from positive to negative at $x = c$, then $f(c)$ is a **local maximum**.
   - If $f'(x)$ changes from negative to positive at $x = c$, then $f(c)$ is a **local minimum**.
   - If $f'(x)$ does not change sign, then $x = c$ is neither a local minimum nor a local maximum.

---

### Fermat's Theorem (Proof)

**Statement**: If $f(x)$ has a local extremum (minimum or maximum) at $x = c$ and $f'(c)$ exists, then $f'(c) = 0$.

#### Proof:
1. Suppose $f(x)$ has a local maximum at $x = c$.
2. By definition, there exists an interval $(a, b)$ such that $f(c) \geq f(x)$ for all $x \in (a, b)$.
3. Consider the derivative definition:
   $$
   f'(c) = \lim_{h \to 0} \frac{f(c + h) - f(c)}{h}
   $$
4. For $h > 0$, $f(c + h) \leq f(c)$, so:
   $$
   \frac{f(c + h) - f(c)}{h} \leq 0
   $$
5. For $h < 0$, $f(c + h) \leq f(c)$, so:
   $$
   \frac{f(c + h) - f(c)}{h} \geq 0
   $$
6. Taking the limit as $h \to 0$ from both sides:
   $$
   \lim_{h \to 0^+} \frac{f(c + h) - f(c)}{h} \leq 0 \quad \text{and} \quad \lim_{h \to 0^-} \frac{f(c + h) - f(c)}{h} \geq 0
   $$
7. Since the derivative exists, the left-hand and right-hand limits are equal:
   $$
   f'(c) = 0
   $$

The same reasoning applies for a local minimum. Hence, Fermat's theorem is proved.

---

## Second Derivative Test

The **second derivative test** provides another way to classify critical points. Critical points are obtained from the **first derivative test**, where $$ f'(x) = 0 $$ or $$ f'(x) $$ is undefined.

### Explanation:
The basis of the second derivative test is that:
1. If $$ f'(x) $$ changes from **positive to negative** at a critical point $$ x = c $$, then $$ f(x) $$ has a **local maximum** at $$ x = c $$.
2. If $$ f'(x) $$ changes from **negative to positive** at a critical point $$ x = c $$, then $$ f(x) $$ has a **local minimum** at $$ x = c $$.

The second derivative $$ f''(x) $$ provides additional information:
- If $$ f''(c) < 0 $$, then $$ f'(x) $$ is **decreasing** at $$ x = c $$, confirming a **local maximum**.
- If $$ f''(c) > 0 $$, then $$ f'(x) $$ is **increasing** at $$ x = c $$, confirming a **local minimum**.
- If $$ f''(c) = 0 $$, the second derivative test is **inconclusive**, and higher-order derivatives may be needed.

### Steps:
1. Find the critical points of $$ f(x) $$ by solving $$ f'(x) = 0 $$ or where $$ f'(x) $$ is undefined.
2. Compute the second derivative $$ f''(x) $$ at each critical point:
   - If $$ f''(c) > 0 $$, then $$ f(c) $$ is a **local minimum** (concave up).
   - If $$ f''(c) < 0 $$, then $$ f(c) $$ is a **local maximum** (concave down).
   - If $$ f''(c) = 0 $$, the test is inconclusive, and higher-order derivatives may be needed.

---

### When Higher-Order Derivatives Are Needed

If $$ f''(c) = 0 $$, the second derivative test does not provide information about the critical point. In such cases, we examine higher-order derivatives.

#### Case 1: Higher-Order Derivative Determines the Nature of the Critical Point
If the $$ n $$-th derivative $$ f^{(n)}(c) \neq 0 $$ for the smallest $$ n > 2 $$:
- If $$ n $$ is **odd**, the critical point is a **point of inflection** (neither a maximum nor a minimum).
- If $$ n $$ is **even**:
  - If $$ f^{(n)}(c) > 0 $$, then $$ f(c) $$ is a **local minimum**.
  - If $$ f^{(n)}(c) < 0 $$, then $$ f(c) $$ is a **local maximum**.

#### Case 2: Example of Higher-Order Derivatives
1. **Example 1: $$ f(x) = x^4 $$**
   - $$ f'(x) = 4x^3 $$, so $$ f'(x) = 0 $$ at $$ x = 0 $$.
   - $$ f''(x) = 12x^2 $$, so $$ f''(0) = 0 $$ (inconclusive).
   - $$ f^{(4)}(x) = 24 $$, so $$ f^{(4)}(0) > 0 $$.
   - Since the fourth derivative is positive, $$ f(x) $$ has a **local minimum** at $$ x = 0 $$.

2. **Example 2: $$ f(x) = x^3 $$**
   - $$ f'(x) = 3x^2 $$, so $$ f'(x) = 0 $$ at $$ x = 0 $$.
   - $$ f''(x) = 6x $$, so $$ f''(0) = 0 $$ (inconclusive).
   - $$ f^{(3)}(x) = 6 $$, so $$ f^{(3)}(0) \neq 0 $$.
   - Since the third derivative is nonzero, $$ x = 0 $$ is a **point of inflection**.

---

### Summary of the Second Derivative Test and Higher-Order Derivatives
- The **second derivative test** is a quick way to classify critical points when $$ f''(x) \neq 0 $$.
- When $$ f''(x) = 0 $$, higher-order derivatives are needed to determine the nature of the critical point.
- Higher-order derivatives provide additional insights into the behavior of the function near the critical point.

---

## Least Squares Optimization

The **least squares method** minimizes the error between observed data points and a model. Assume we have a single-dimensional input $x$ and a target output $y$, and we optimize the least squares error.

### Objective Function  

The least squares objective function is:  

$$
\min_w L(w) = \sum_{i=1}^n \left( y_i - f(x_i; w) \right)^2
$$

For a linear model $f(x; w) = wx$, the objective simplifies to:  

$$
\min_w L(w) = \sum_{i=1}^n \left( y_i - wx_i \right)^2
$$

### First Derivative
To find the optimal $w$, we compute:

$$
\begin{aligned}
\frac{dL(w)}{dw} &= \frac{d}{dw} \sum_{i=1}^n \left( y_i - wx_i \right)^2 \\
&= \sum_{i=1}^n 2 \left( y_i - wx_i \right) (-x_i) \\
&= -2 \sum_{i=1}^n x_i \left( y_i - wx_i \right) \\
&= -2 \sum_{i=1}^n x_i y_i + 2w \sum_{i=1}^n x_i^2 \\
&= 2w \sum_{i=1}^n x_i^2 - 2 \sum_{i=1}^n x_i y_i
\end{aligned}
$$

### Critical Point
Setting the first derivative to zero:

$$
\begin{aligned}
2w \sum_{i=1}^n x_i^2 - 2 \sum_{i=1}^n x_i y_i &= 0 \\
w &= \frac{\sum_{i=1}^n x_i y_i}{\sum_{i=1}^n x_i^2}
\end{aligned}
$$

### Second Derivative
$$
\begin{aligned}
\frac{d^2L(w)}{dw^2} &= \frac{d}{dw} \left( 2w \sum_{i=1}^n x_i^2 - 2 \sum_{i=1}^n x_i y_i \right) \\
&= 2 \sum_{i=1}^n x_i^2
\end{aligned}
$$

Since $\sum_{i=1}^n x_i^2 \geq 0$, the second derivative is positive, confirming that $w^*$ is a **minimum**.

### Optimal Point
$$
w^* = \frac{\sum_{i=1}^n x_i y_i}{\sum_{i=1}^n x_i^2}
$$

---
### Optimal Loss Value
Substituting $w^*$ into the loss function:

$$
\begin{aligned}
L(w^*) &= \sum_{i=1}^n \left( y_i - w^* x_i \right)^2 \\
&= \sum_{i=1}^n \left( y_i - \frac{\sum_{j=1}^n x_j y_j}{\sum_{j=1}^n x_j^2} x_i \right)^2
\end{aligned}
$$

Expanding:

$$
\begin{aligned}
L(w^*) &= \sum_{i=1}^n \left( y_i^2 - 2 y_i \cdot \frac{\sum_{j=1}^n x_j y_j}{\sum_{j=1}^n x_j^2} x_i + \left( \frac{\sum_{j=1}^n x_j y_j}{\sum_{j=1}^n x_j^2} x_i \right)^2 \right)
\end{aligned}
$$

Splitting the summation:

$$
\begin{aligned}
L(w^*) &= \sum_{i=1}^n y_i^2 - 2 \frac{\sum_{j=1}^n x_j y_j}{\sum_{j=1}^n x_j^2} \sum_{i=1}^n x_i y_i + \sum_{i=1}^n \left( \frac{\sum_{j=1}^n x_j y_j}{\sum_{j=1}^n x_j^2} x_i \right)^2
\end{aligned}
$$

Factoring out the squared term:

$$
\begin{aligned}
\sum_{i=1}^n \left( \frac{\sum_{j=1}^n x_j y_j}{\sum_{j=1}^n x_j^2} x_i \right)^2 &= \frac{\left(\sum_{j=1}^n x_j y_j\right)^2}{\left(\sum_{j=1}^n x_j^2\right)^2} \sum_{i=1}^n x_i^2
\end{aligned}
$$

Since $\sum_{i=1}^n x_i^2$ appears in the denominator:

$$
\begin{aligned}
L(w^*) &= \sum_{i=1}^n y_i^2 - \frac{\left( \sum_{i=1}^n x_i y_i \right)^2}{\sum_{i=1}^n x_i^2}
\end{aligned}
$$

This represents the minimized loss after optimizing $w$. It shows how much error remains after the best-fit parameter is chosen.
### Summary
- The first derivative gives the critical point $w^*$.
- The second derivative confirms that $w^*$ is a **minimum** because it is positive.
- The least squares method minimizes the error between the observed and predicted values.

-------------
## Some of the applications of derivatives
## 1. Finding Minima and Maxima

Derivatives are widely used to optimize functions by locating their minimum or maximum values. 

### Example: Minimize the Perimeter of a Rectangle Given Its Area

- **Problem Statement**: A rectangle has a fixed area of 100 square units. Find the dimensions that minimize its perimeter.
- **Step-by-Step Solution**:
  1. **Define Variables and Functions**:
     - Let $l$ be the length and $w$ be the width.
     - Area: $l \cdot w = 100$.
     - Perimeter: $P = 2l + 2w$.
  2. **Express Perimeter in One Variable**:
     - Solve for $w$ from the area:  
       $$ w = \frac{100}{l} $$
     - Substitute into the perimeter:  
       $$ P(l) = 2l + 2 \cdot \frac{100}{l} = 2l + \frac{200}{l} $$
  3. **Take the Derivative**:
     - Compute $P'(l)$:  
       $$ P'(l) = \frac{d}{dl} \left( 2l + \frac{200}{l} \right) = 2 - \frac{200}{l^2} $$
  4. **Find Critical Points**:
     - Set the derivative to zero:  
       $$ 2 - \frac{200}{l^2} = 0 $$
       $$ \frac{200}{l^2} = 2 $$
       $$ l^2 = 100 $$
       $$ l = 10 \quad (\text{since } l > 0) $$
  5. **Find the Width**:
     - $$ w = \frac{100}{l} = \frac{100}{10} = 10 $$
  6. **Verify Minimum**:
     - Second derivative:  
       $$ P''(l) = \frac{d}{dl} \left( 2 - \frac{200}{l^2} \right) = \frac{400}{l^3} $$
     - At $l = 10$:  
       $$ P''(10) = \frac{400}{10^3} = 0.4 > 0 $$  
       A positive second derivative confirms a local minimum.
  7. **Calculate Perimeter**:
     - $ P = 2l + 2w = 2 \cdot 10 + 2 \cdot 10 = 40 $

- **Conclusion**: The minimum perimeter is 40 units when $l = w = 10$, forming a square. 
---

## 2. Finding Roots of $f(x) = 0$ Using Newton’s Method

Newton’s method leverages derivatives to approximate roots of equations numerically. It’s one of the  method in optimization for solving nonlinear equations iteratively.

### Detailed Explanation

- **Concept**: Start with an initial guess and refine it using the tangent line’s slope (the derivative) to approach the root.
- **Formula**:  
  $$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $$  
  where $f'(x_n)$ is the derivative at $x_n$.

### Example: Find the Root of $x^2 - 2 = 0$ (i.e., $\sqrt{2}$)

- **Function**: $f(x) = x^2 - 2$.
- **Derivative**: $f'(x) = 2x$.
- **Step-by-Step Solution**:
  1. **Initial Guess**: Choose $x_0 = 1$ (close to $\sqrt{2} \approx 1.414$).
  2. **First Iteration**:
     - $$ f(x_0) = 1^2 - 2 = -1 $$
     - $$ f'(x_0) = 2 \cdot 1 = 2 $$
     - $$ x_1 = x_0 - \frac{f(x_0)}{f'(x_0)} = 1 - \frac{-1}{2} = 1 + 0.5 = 1.5 $$
  3. **Second Iteration**:
     - $$ f(x_1) = 1.5^2 - 2 = 2.25 - 2 = 0.25 $$
     - $$ f'(x_1) = 2 \cdot 1.5 = 3 $$
     - $$ x_2 = 1.5 - \frac{0.25}{3} = 1.5 - 0.08333 \approx 1.41667 $$
  4. **Third Iteration**:
     - $$ f(x_2) = 1.41667^2 - 2 \approx 2.00694 - 2 = 0.00694 $$
     - $$ f'(x_2) = 2 \cdot 1.41667 \approx 2.83334 $$
     - $$ x_3 = 1.41667 - \frac{0.00694}{2.83334} \approx 1.41667 - 0.00245 \approx 1.41422 $$
  5. **Check Convergence**:
     - $$ f(1.41422) \approx 1.41422^2 - 2 \approx 0.00002 $$  
       The result is nearly zero, showing convergence to $\sqrt{2}$.

- **Conclusion**: After three iterations, $x \approx 1.41421$, closely matching $\sqrt{2}$. 

---

## 3. Linear Approximations

Derivatives enable linear approximations, approximating function values near a known point using the tangent line. This is useful for estimating values like square roots without complex computations.

### Detailed Explanation

- **Formula**: For a function $f(x)$, the linear approximation near $x = a$ is:  
  $$ f(x) \approx f(a) + f'(a)(x - a) $$
- **Application**: Approximate square roots using the nearest perfect square.

### Example: Approximate $\sqrt{50}$ Using $\sqrt{49}$

- **Function**: $f(x) = \sqrt{x}$.
- **Derivative**: $f'(x) = \frac{1}{2\sqrt{x}}$.
- **Step-by-Step Solution**:
  1. **Choose a Nearby Point**:
     - Use $a = 49$, since $49$ is the nearest perfect square to 50.
  2. **Evaluate Function and Derivative**:
     - $$ f(49) = \sqrt{49} = 7 $$
     - $$ f'(49) = \frac{1}{2\sqrt{49}} = \frac{1}{2 \cdot 7} = \frac{1}{14} $$
  3. **Apply Linear Approximation**:
     - For $x = 50$:  
       $$ f(50) \approx f(49) + f'(49)(50 - 49) $$
       $$ f(50) \approx 7 + \frac{1}{14} \cdot 1 = 7 + \frac{1}{14} \approx 7 + 0.07143 = 7.07143 $$
  4. **Verify Accuracy**:
     - Actual value: $\sqrt{50} \approx 7.07107$.
     - Error: $7.07143 - 7.07107 = 0.00036$, a small difference.

- **Conclusion**: The linear approximation gives $\sqrt{50} \approx 7.07143$, very close to the actual value. 

----------------
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


----