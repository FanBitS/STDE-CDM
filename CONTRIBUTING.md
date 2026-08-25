# Contributing

Bug reports and reproducibility questions are welcome through GitHub Issues.
When reporting a problem, include the command, configuration, Python and
PyTorch versions, hardware, complete traceback, and the smallest reproducible
example.

For code contributions:

1. Create a branch from `main`.
2. Keep changes focused and preserve the locked data and evaluation protocol.
3. Run `make compile` and `make test`.
4. Document any new dependency, command-line option, or result schema.
5. Open a pull request describing the scientific and software impact.

Do not commit Gurobi license files, credentials, private datasets, or
intermediate training outputs.
