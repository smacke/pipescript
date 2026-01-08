pipescript
==========

[![CI Status](https://github.com/smacke/pipescript/workflows/pipescript/badge.svg)](https://github.com/smacke/pipescript/actions)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![License: BSD3](https://img.shields.io/badge/License-BSD3-maroon.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Versions](https://img.shields.io/pypi/pyversions/pipescript.svg)](https://pypi.org/project/pipescript)
[![PyPI Version](https://img.shields.io/pypi/v/pipescript.svg)](https://pypi.org/project/pipescript)

Pipescript is an IPython extension that brings a pipe operator `|>` and
powerful placeholder and macro expansion syntax extensions to IPython and Jupyter.

If you're familiar with the [magrittr](https://magrittr.tidyverse.org/) package
for R, then you'll be right at home with pipescript.


## Getting Started

Run the following in IPython or Jupyter to install pipescript and load
the extension:

```python
%pip install pipescript
%load_ext pipescript
```

The `%load_ext pipescript` invocation is what enables the new pipe syntax
in your current session.

## Features by Example

Let's look at a few examples to give a flavor of what you can do with pipescript:

```python
# Render a sorted version of a tuple
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
function construction: for pipescript, we use `$` to stand in for function arguments
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
pipescript permits *named placeholders* by prefixing `$` to an identifier:

```python
# Pair even entries from a range with their adjacent odd entry
range(6) |> list |> zip($v[::2], $v[1::2]) |> list
>>> [(0, 1), (2, 3), (4, 5)]
```

In the above example, we could have used any name for `$v`, the important
thing is that the same name was used -- otherwise pipescript would have
induced a function with two arguments instead of one.

### Undetermined Pipelines

Similar to magrittr's behavior, if any number of placeholders appear in the first
step of an pipescript pipeline, this *undetermined pipeline* will represent a function:

```python
>>> second_largest_value = $ |> sorted($, reverse=True) |> $[1]
>>> [3, 8, 6, 5, 1] |> second_largest_value
6
```

### Macros and Partial Function Syntax

In some cases, it may be desirable to curry a function with parameters at its start,
akin to the typical usage of `functools.partial`. For example:

```python
>>> add_reducer = reduce(lambda x, y: x + y, $, $)
>>> add_reducer([1, 2, 3], 0)
6
>>> add_reducer([[1, 2, 3], [4, 5, 6]], [])
[1, 2, 3, 4, 5, 6]
```

To avoid writing out a `$` placeholder for each and every tail argument, you can
prefix the call itself with a `$` and omit subsequent arguments, just like in coconut:

```python
>>> add_reducer = reduce$(lambda x, y: x + y)
>>> add_reducer([1, 2, 3], 0)
6
>>> add_reducer([[1, 2, 3], [4, 5, 6]], [])
[1, 2, 3, 4, 5, 6]
```

Or even more simply, since the induced partial function retains all the same
argument defaults as the original `reduce`, we can omit the base case:

```python
>>> add_reducer = reduce$(lambda x, y: x + y)
>>> add_reducer([1, 2, 3])
6
>>> add_reducer([[1, 2, 3], [4, 5, 6]])
[1, 2, 3, 4, 5, 6]
```

For common functional programming tools like `map`, `reduce`, and `filter`, the above
pattern is so common that pipescript provides corresponding macros, in which the function used
to curry each higher order function is specified between brackets:

```python
>>> add_reducer = reduce[lambda x, y: x + y]
>>> [1, 2, 3] |> add_reducer
6
>>> [[1, 2, 3], [4, 5, 6]] |> add_reducer
[1, 2, 3, 4, 5, 6]
```

We're still writing out `lambda x, y: x + y`, which is kind of tedious -- for these
kinds of simple lambda constructions, pipescript provides a *quick lambda macro*, `f`:

```python
>>> add_reducer = reduce[f[$ + $]]
>>> [1, 2, 3] |> add_reducer
6
>>> [[1, 2, 3], [4, 5, 6]] |> add_reducer
[1, 2, 3, 4, 5, 6]
```

`f` can also be used on its own:

```python
>>> f[$ + $](2, 3)
5

>>> f[$a*$b + $b*$c + $a*$c](2, 3, 4)
26
```

Furthermore, pipescript allows you to omit the `f` from higher order
functional macros, so that you can simply do `add_reducer = reduce[$ + $]` instead.
Here are a couple of nifty constructions utilizing this compact syntax:

```python
# factorial
>>> range(1, 5) |> reduce[$ * $]
24

# compute a number from decimal digits
>>> [2, 3, 4] |> reduce[10*$ + $]
234
```

### Additional Pipe Operators

There are a few other variants of the `|>` operator offered by
pipescript, covered in this section.

#### Assignment Pipe

The *assignment pipe*, `|>>`, writes the left hand side value to the variable
whose name is specified on the right hand side. Furthermore, it evaluates to
the left hand side value. For example:

```python
>>> 2 |> $ + 2 |>> two_plus_two |> $ + 3 |>> two_plus_two_plus_three
7
>>> (two_plus_two, two_plus_two_plus_three)
(4, 7)
```

#### Varargs Pipe

The *varargs pipe*, `*|>`, unpacks the iterable on the left hand side before
passing its values as inputs to the function on the right hand side. For
example:

```python
# Add two numbers:
>>> (2, 3) *|> $ + $
5
```

A common pattern is using `*|>` to expand an undetermined pipeline
appearing inside of a `map[...]`:

```python
# Take the product of consecutive pairs of even-odd integers
>>> consecutive_pairs = range(10) |> list |> ($v[::2], $v[1::2]) *|> zip
>>> consecutive_pairs |> map[$ *|> $ * $] |> list
[0, 6, 20, 42, 72]
```

#### Function Pipe

The other commonly used pipe is the *function pipe*, `.>`, which is used to compose
the functions specified on the left hand side and right hand side together, with the
function on the left hand side being applied first in the composition (note that this
behavior is reversed from normal function composition, but follows the flow of data better).
For example:

```python
>>> reverse = reversed .> list
>>> [1, 2, 3] |> reverse
[3, 2, 1]
```

#### Other Pipes

Besides `|>>`, `*|>`, and `.>`, pipescript offers a few less commonly used operators as well. The below
table describes the complete set of forward pipe operators available:

| Operator           | Pipescript Syntax                                   | Python Syntax                           |
|--------------------|-----------------------------------------------------|-----------------------------------------|
| <code>\|></code>   | <code>y = x \|> f</code>                            | `y = f(x)`                              |
| <code>\|>></code>  | <code>x \|>> y</code>                               | `y = x; y`                              |
| <code>*\|></code>  | <code>y = x *\|> f</code> where `x` is an iterable  | `y = f(*x)`                             |
| <code>**\|></code> | <code>y = x **\|> f</code> where `x` is a dict      | `y = f(**x)`                            |
| `.>`               | `h = g .> f`                                        | `h = lambda *a, **kw: g(f(*a, **kw))`   |
| `*.>`              | `h = g *.> f`                                       | `h = lambda *a, **kw: g(*f(*a, **kw))`  |
| `**.>`             | `h = g **.> f`                                      | `h = lambda *a, **kw: g(**f(*a, **kw))` |
| `?>`               | `y = x ?> f`                                        | `y = None if x is None else f(x)`       |
| `*?>`              | `y = x *?> f` where `x` is an iterable, or `None`   | `y = None if x is None else f(*x)`      |
| `**?>`             | `y = x **?> f` where `x` is a dict, or `None`       | `y = None if x is None else f(**x)`     |
| `$>`               | `g = x $> f`                                        | `g = functools.partial(f, x)`           |
| `*$>`              | `g = x *$> f` where `x` is an iterable              | `g = functools.partial(f, *x)`          |
| `**$>`             | `g = x **$> f` where `x` is a dict                  | `g = functools.partial(f, **x)`         |

Except for `|>>`, each and every operator has a corresponding *backward* variant; e.g. `<|` is the backward variant
of `|>` and is a low-precedence apply. For example:

```python
>>> reversed .> list <| [1, 2, 3]
[3, 2, 1]
```

All pipe operators are applied in order from left to right (including backward pipes).
Furthermore, all pipe operators are left associative and operate at the same precedence
as `|` (bitwise or), meaning that any pipeline steps that include an `|` binary operation
must be wrapped in parentheses.

### Additional Macros and Helper Utilities

#### `do` macro

Similar to [toolz](https://github.com/pytoolz/toolz), pipescript offers a `do` macro
implementing something similar to the following higher order function:

```python
def do(func, obj):
    func(obj)
    return obj
```

In the case of pipescript, the input function `func` is specified inside of brackets,
just as with other functional macros:

```python
>>> 2 |> $ + 2 |> do[print] |> $ + 2 |>> result
4
6
```

While any function expression, including undetermined pipelines, can appear inside `do[...]` brackets,
`do[print]` is so common that pipescript provides a `peek` utility that implements the very same:

```python
>>> 2 |> $ + 2 |> peek |> $ + 2 |>> result
4
6
```

To suppress the automatic expression rendering of a pipeline result, pipescript also offers a `null` utility function
(as in `/dev/null`), which essentially swallows its input:

```python
>>> 2 |> $ + 2 |> peek |> $ + 2 |>> result |> null
4
```

#### `fork` and `parallel` macros

If you wish to move beyond linear chains and apply the same input to multiple pipelines,
pipescript provides `fork` and `parallel` macros, which return the results of each function
as a tuple:

```python
>>> range(10) |> list |> fork[
    map[2 * $] .> filter[$ % 3 == 0],
    map[3 * $] .> filter[$ % 2 == 0],
]
([0, 6, 12, 18], [0, 6, 12, 18, 24])
```

`parallel` does the same thing as `fork` but executes each function passed to it concurrently.

#### `when` `unless`, `otherwise`, `repeat`, `until` macros

The `when` macro takes as input a value and conditional expression that, upon passing,
forwards the value, and upon failing, terminates computation with `None`. It is particularly powerful
when combined with `fork` and `collapse` (the latter of which extracts the non-null value out of
the tuple that results from the `fork`):

```python
>>> collatz = when[$ != 1] .> fork[
    when[$ % 2 == 0] .> $ // 2,
    when[$ % 2 == 1] .> $ * 3 + 1,
] .> collapse .> peek
```

You can also use `unless`, which is just the opposite of `when`:

```python
>>> collatz = when[$ != 1] .> fork[
    when[$ % 2 == 0] .> $ // 2,
    unless[$ % 2 == 0] .> $ * 3 + 1,
] .> collapse .> peek
```

If you don't want to explicitly write out the negative conditional, `fork` lets you
use the `otherwise` macro as the last expression:

```python
>>> collatz = when[$ != 1] .> fork[
    when[$ % 2 == 0] .> $ // 2,
    otherwise[$ * 3 + 1],
] .> collapse .> peek
```

Of course, this can be written more naturally and succinctly with
a ternary conditional expression:

```python
>>> collatz = when[$ != 1] .> f[$v // 2 if $v % 2 == 0 else $v * 3 + 1] .> peek
```

Regardless of how we write the conditional, pipescript allows you to
exponentiate single-argument functions with power the composition (`.**`)
operator, so that we don't need to write out
`42 |> collatz |> collatz |> ... |> collatz`:

```python
>>> 42 |> collatz .** 20
21
64
32
16
8
4
2
1
```

If you don't want to guess the upper bound of how many steps to run it, you can
use the `repeat` and `until` macros (`until` is just an alias of `unless`):

```python
>>> collatz = f[$v // 2 if $v % 2 == 0 else $v * 3 + 1]
>>> 42 |> repeat[until[$ == 1] .> collatz .> peek] |> null
21
64
32
16
8
4
2
1
```

#### `future` macro

Finally, to schedule a function to run in another thread and immediately
return a future to the eventual result, pipescript provides a `future` macro:

```python
>>> 2 |> future[$ + 2] |> $.result()
4
>>> [1, 2, 3] |> future[sum] |> $.result()
6
```

## Placeholder Scope

A natural question is: how does pipescript know what part of the code should
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

## Optional Chaining, Permissive Attribute Chaining, and Nullish Coalescing

Pipescript also provides typescript-style optional chaining and nullish coalescing.
That is, `a?.b.c.d().e` resolves to `None` when `a` is `None`, as does `a?.()`.
Also, `a ?? obj` evaluates to `obj` only when `a` is `None`, but evaluates to `a`
whenever `a` is some other falsey value like `""`, `0`, `False`, or `[]`. Note that,
like normal boolean `or`, the nullish coalescing operator `??` is lazy and will not
evaluate expressions on its right hand side when its left hand side is not `None`.

Unlike Javascript, Python does not resolve unavailable attribute accesses to
`undefined`, but will rather throw `AttributeError`. In pipescript, if you would
like to perform some kind of permissive attribute access like in Javascript, you
can use the *permissive chaining operator* `.?` (where the `?` appears after the
`.`) and access `b` as `a.?b`, which is equivalent to `getattr(a, "b", None)`.
Note however that if the aforementioned expression resolves to `None`, something
like `a.?b.c` will still throw an `AttributeError` -- to avoid that, you need to
combine both permissive attribute chaining and optional chaining as `a.?b?.c`.

## Performance Overhead

Because pipescript is implemented using instrumentation (see [How it works](#how-it-works)),
it does incur overhead. For top-level code written in a Jupyter cell (e.g.,
code that doesn't have any indentation), the additional overhead generally doesn't matter,
as it tends to be insignificant when compared to data-intensive dataframe operations
and SQL queries common in data science workloads. Furthermore, overhead is only incurred
when pipescript syntax is actually used -- there's no penalty for any code written in vanilla
Python, **even when pipescript has been enabled in your current REPL session**.

## More Examples
I developed pipescript while working on
[Advent of Code 2025](https://adventofcode.com/2025) in parallel,
and used it for most of the input processesing portions of my solutions.
You can find these solutions at https://github.com/smacke/aoc2025. In particular,
the [solution for day 6](https://github.com/smacke/aoc2025/blob/main/aoc6.ipynb)
showcases the upper limits of what is possible with pipescript. Note however that it is
optimized for pipescript usage and not readability, which I generally wouldn't recommend.

## What pipescript is and is not

For now, pipescript is not a general purpose functional programming language on top of
Python. It is very much not intended for production use cases, and instead
caters toward quick-and-dirty one-off / scratchpad type computations in IPython
and Jupyter specifically. In short, pipescript aims to provide simple but powerful
pipeline and placeholder syntax to interactive Python programming environments.

Particularly, pipescript is:
- Currently only for interactive Python environments built on top of IPython, such as
  Jupyter, or IPython itself
- Just a library you can install from PyPI, compatible with a wide range of Python 3
  versions -- no fancy installation instructions, no complicated language distribution
  to install
- Fully compatible with all existing Python standard and third-party libraries that
  you already know and love, since it's just Python function calls under the hood

All the different pipeline operators like `|>`, `<|`, `*|>`, etc. essentially
transpile down to an instrumented variant of the bitwise-or (`|`) operator, and
therefore every new operator left-associates at the same level of precedence,
meaning that pipeline steps run from left to right in the order that they
appear. Pipescript aims to optimize for simplicity, readability / writability, and
predictability over feature completeness (though I'd like to think it strikes a
fairly good balance in this regard). Pipescript may be expanded beyond IPython / Jupyter
depending on traction.

## How it works

Pipescript works by transforming syntax in two stages. First, it rewrites token spans
like `|>` and `*|>` that are illegal in Python to legal ones -- for the previous
examples, both spans are rewritten to bitwise or, `|`. After these transformations,
the resulting code is valid (but likely not runnable) Python syntax. Pipescript uses
the [pyccolo](https://github.com/smacke/pyccolo) library to perform these rewrites,
which remembers the positions of the rewrites where they occurred, so that the eventual
`ast.BinOp` AST node can be associated with the `|>` operator.

Pyccolo is a library I developed during my PhD which provides an event-driven
architecture for declarative AST transformations. Its key selling point is that
it allows you to layer multiple AST transformations on top of each other in a
composable fashion. In short, you specify handlers for different AST nodes such
as `ast.BinOp`, and pyccolo instruments these nodes by emitting events for them,
so that when the code runs, all the handlers for a particular event are run.
Such event handlers are what allow us to change the behavior of `ast.BinOp`
nodes that have been associated with various custom operators like `|>`.

Because the same event emission transformation can be leveraged by multiple
associated handlers, you generally don't need to worry about said
transformations rewriting the AST in ways that conflict with each other. This
composability lies in stark contrast with the challenges you would face if you
were to just create a bunch of `ast.NodeTransformer` instances to perform
transformations. The strategy employed by pyccolo therefore allows for
incremental and iterative feature development without requiring large rewrites
as new features are introduced.

To summarize, pipescript rewrites its syntax to valid Python, and then runs this Python in
an instrumented fashion using pyccolo. Because everything is just running in
Python, pipescript is effectively a Python superset, and because the transformed
Python that is instrumented is fairly similar visually to pipescript syntax,
various Jupyter ergonomical features like readable stack traces and jedi-based
autocomplete can continue to function as normal (for the most part).

Implementation-wise, thanks to pyccolo's heavy lifting, I was able to write the
initial release of pipescript entirely over the course of time off during the
2025 holiday season. At the time of this writing, pipescript occupies about 2000
lines of code (excluding tests), each of which was produced *without* the help
of any AI agents.

## Inspiration

Pipescript draws inspiration largely from
[magrittr](https://magrittr.tidyverse.org/), but also from efforts like
[coconut](https://coconut-lang.org/) (a functional superset of Python),
as well as from libraries like [Pipe](https://github.com/JulienPalard/Pipe) and [toolz](https://github.com/pytoolz/toolz) which
fill some of Python's pipe and functional programming gaps with elegant APIs.

## Disclaimer

**Warning: use pipescript at your own risk!** It is very much not guaranteed to
be bug-free -- I implemented it in a hurry before it was time to go back to work.

## License
Code in this project licensed under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
