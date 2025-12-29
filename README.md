nbpipes
=======

[![CI Status](https://github.com/smacke/nbpipes/workflows/nbpipes/badge.svg)](https://github.com/smacke/nbpipes/actions)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![License: BSD3](https://img.shields.io/badge/License-BSD3-maroon.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Versions](https://img.shields.io/pypi/pyversions/nbpipes.svg)](https://pypi.org/project/nbpipes)
[![PyPI Version](https://img.shields.io/pypi/v/nbpipes.svg)](https://pypi.org/project/nbpipes)

nbpipes is an IPython extension that brings a pipe operator `|>` and
powerful placeholder syntax extensions to IPython and Jupyter. It is:
- Just a library you can install from PyPI, compatible with a wide range of Python 3
  versions -- no fancy installation instructions, no complicated language distribution
  to install
- Intended for Jupyter notebooks, for the IPython REPL, or for any interactive
  Python environment built on top of IPython
- Fully compatible with all existing Python standard and third-party libraries that
  you already know and love

If you're familiar with the [magrittr](https://magrittr.tidyverse.org/) package
for R, then you'll be right  at home with nbpipes.


## Getting Started

Run the following in IPython or Jupyter to install nbpipes and load
the extension:

```python
%pip install nbpipes
%load_ext nbpipes
```

The `%load_ext nbpipes` invocation is what enables the new pipe syntax
in your current session.

## Features by Example

Let's look at a few examples to give a flavor of what you can do with nbpipes:

```python
# Display a sorted version of a tuple
>>> tup = (3, 4, 1, 5, 6)
>>> tup |> sorted |> tuple
(1, 3, 4, 5, 6)
```
The above example showcases the `|>`, or "pipe", operator, which is a much-loved
feature of functional programming that has become increasingly mainstream. Its
primary benefit is that the flow of execution follows natural left-to-right
reading / writing order of the code. Whether or not such pipeline syntax is
available, it's not uncommon for programmers to execute pipelines like the above
multiple times during to verify the computation at each step, particularly in
interactive programming environments like Jupyter. With `|>`, this type of
incremental verification becomes a breeze: first execute `tup |> sorted`, then
append ` |> tuple` to execute the full chain `tup |> sorted |> tuple`, each time
using the last-expression rendering capabilities of the notebook or REPL to
inspect and verify the result.

### Placeholders

The power of the `|>` operator is amplified via placeholder syntax for implicit
lambda construction: for nbpipes, we use `$` to stand in for function arguments
and induce function creation:

```python
# Sort a list in reverse order
>>> lst = [3, 4, 1, 5, 6]
>>> lst |> sorted($, reverse=True)
[6, 5, 4, 3, 1]
```

`$` is analogous to magrittr's `.` placeholder. It can also be used outside
of pipeline contexts:

```python
# Sort a list in reverse order and print the result
lst = [3, 4, 1, 5, 6]
reverse_sorter = sorted($, reverse=True)

# The following are equivalent:
print(reverse_sorter(lst))
lst |> reverse_sorter |> print
```

Each time `$` appears, it represents a new argument, so `sorted($, reverse=$)`
represents a function with two arguments:

```python
import random

# Sort a list in either ascending or descending order with probablility 0.5:
lst = [3, 4, 1, 5, 6]
sorter = sorted($, reverse=$)
reverse = random.random() < 0.5

# The following are equivalent:
print(sorter(lst, reverse))
lst |> sorter($, reverse) |> print
```

Placeholders can appear anywhere -- not just as arguments to function calls:

```python
# Sort a list and find the position of element 4:
>>> lst = [3, 4, 1, 5, 6]
>>> lst |> sorted |> $.index(3)
1
```

### Named Placeholders

There are situations that would benefit from referencing the same placeholder multiple times, for which
nbpipes permits *named placeholders* by prefixing `$` to an identifier:

```python
# Pair even entries from a range with their adjacent odd entry
range(6) |> list |> zip($v[::2], $v[1::2]) |> list
>>> [(0, 1), (2, 3), (4, 5)]
```

In the above example, we could have used any name for `$v`, the important
thing is that the same name was used -- otherwise nbpipes would have
induced a function with two arguments instead of one.

### Undetermined Pipelines

Similar to magrittr's behavior, if any number of placeholders appear in the first
step of an nbpipes pipeline, this *undetermined pipeline* will represent a function:

```python
>>> second_largest_value = $ |> sorted($, reverse=True) |> $[1]
>>> [3, 8, 1, 5, 6] |> second_largest_value
6
```

### Macros and Curry Syntax

### Helper Utilities

### Additional Operators

## Placeholder Scope

A natural question is: how does nbpipes know what part of the code should
be included in the body of the function induced by placeholder use? The
rules are as follows:

1. If there is a macro or pipeline step enclosing the placeholder, the induced
   function body includes the "smallest" such enclosing macro or pipeline step.
2. Otherwise, the function body expands to include the nearest "chain"
   of function calls, attribute accesses, and / or subscript accesses.

An example of a "chain" would be something like `np.array($).T.astype(int)`,
which induces a lambda that converts its argument to a numpy array,
transposes it, and then converts the result to use `int64` dtype. That is,
the lambda body expands to include not just `np.array($)`, but the entire
"chain" in the expression.

To see a concrete example of where this matters, consider the following
two placeholder expressions:

```python
# The following sorters do different things!
sorter1 = sorted($, key=$[1])
sorter2 = sorted($, key=f[$[1]])
```

`sorter1` is a function that takes two arguments: a sequence, and a list of
functions, the second of which will be used to compute the sort key, which it then
uses to sort the first argument.
`sorter2`, on the other hand, is a function that takes a single argument, which
is a sequence that it sorts using the second element of each value in said
sequence value as sort key. In most cases, `sorter2` probably gives the desired
behavior.

## Performance Overhead

## More Examples
I developed nbpipes while working on
[Advent of Code 2025](https://adventofcode.com/2025) in parallel,
and used it for most of the input processesing portions of my solutions,
which you can find at https://github.com/smacke/aoc2025.

## What nbpipes is and is not

nbpipes is not a general purpose functional programming language on top of
Python. It is very much not intended for production use cases, and instead
caters toward quick-and-dirty one-off / scratchpad type computations in IPython
and Jupyter specifically. In short, nbpipes aims to provide simple but powerful
pipeline and placeholder syntax to interactive Python programming environments.

All the different pipeline operators like `|>`, `<|`, `*|>`, etc. essentially
transpile down to an instrumented variant of the bitwise-or (`|`) operator, and
therefore every new operator left-associates at the same level of precedence,
meaning that pipeline steps run from left to right in the order that they
appear. nbpipes aims to optimize for simplicity, readability / writability, and
predictability over feature completeness (though I'd like to think it strikes a
fairly good balance in this regard).

## How it works

## Inspiration

nbpipes draws inspiration largely from
[magrittr](https://magrittr.tidyverse.org/), but also from efforts like
[coconut](https://coconut-lang.org/) (a functional superset of Python),
as well as from libraries like [Pipe](https://github.com/JulienPalard/Pipe) which
take a different approach to fill Python's pipe gap with operator overloading hacks.

## License
Code in this project licensed under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
