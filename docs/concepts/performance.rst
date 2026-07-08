Performance
===========

Because pipescript is implemented with instrumentation (see
:doc:`how_it_works`), using its syntax incurs some run-time overhead. Two facts
keep this from mattering in practice.

**Overhead is only paid when pipescript syntax is used.** Vanilla Python runs at
full speed even with the extension loaded in your session -- there is no penalty
for code that does not use any pipescript operators or macros.

**For top-level notebook code, the overhead is generally insignificant.**
pipescript targets interactive, scratchpad-style work, where a pipeline's cost is
dwarfed by the data-intensive dataframe operations and SQL queries typical of
data-science workloads. For code written directly in a Jupyter cell (without deep
indentation), the added overhead tends not to be noticeable.

If you are writing performance-critical inner loops, that is exactly the kind of
production code pipescript is *not* aimed at -- reach for plain Python there. See
:doc:`/getting_started/installation` for the intended scope.
