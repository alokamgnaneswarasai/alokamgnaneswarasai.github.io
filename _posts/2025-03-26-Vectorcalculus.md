---
layout: post
title: Introduction to multi variate calculus
tags: [calculus,Multi Variate, Hessian,gradient]
author: Gnaneswara Sai
---
<!-- {%- include mathjax.html -%} -->


<!--more-->
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
      Using the first principle:
      $$
      \frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = \lim_{h \to 0} \frac{\frac{f(x + h)}{g(x + h)} - \frac{f(x)}{g(x)}}{h}
      $$
      Combine fractions:
      $$
      = \lim_{h \to 0} \frac{\frac{f(x + h) g(x) - f(x) g(x + h)}{g(x + h) g(x)}}{h}
      $$
      Simplify:
      $$
      = \lim_{h \to 0} \frac{f(x + h) g(x) - f(x) g(x + h)}{h \cdot g(x + h) \cdot g(x)}
      $$
      Expand \( f(x + h) \) and \( g(x + h) \) using their Taylor expansions:
      $$
      f(x + h) = f(x) + h f'(x) + \dots, \quad g(x + h) = g(x) + h g'(x) + \dots
      $$
      Substitute into the numerator:
      $$
      f(x + h) g(x) - f(x) g(x + h) = \left[ f(x) + h f'(x) \right] g(x) - f(x) \left[ g(x) + h g'(x) \right]
      $$
      Expand terms:
      $$
      = f(x) g(x) + h f'(x) g(x) - f(x) g(x) - h f(x) g'(x)
      $$
      Simplify:
      $$
      = h \left[ f'(x) g(x) - f(x) g'(x) \right]
      $$
      Substitute back into the limit:
      $$
      \frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) = \lim_{h \to 0} \frac{h \left[ f'(x) g(x) - f(x) g'(x) \right]}{h \cdot g(x + h) \cdot g(x)}
      $$
      Cancel \( h \) in the numerator and denominator:
      $$
      = \lim_{h \to 0} \frac{f'(x) g(x) - f(x) g'(x)}{g(x + h) \cdot g(x)}
      $$
      As \( h \to 0 \), \( g(x + h) \to g(x) \):
      $$
      = \frac{f'(x) g(x) - f(x) g'(x)}{g(x)^2}
      $$
      Thus, the quotient rule is proved:
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
