# General
(Update May 11 2025)

# TODOS

## Next steps
1. More gradient descent testing/debugging, particularly for power cone.
2. Implement the adjoint for QCPs (just need to consider adding in sparsity to already implemented atom). **Quill**
3. Add exponential cone info to paper appendix. **Quill**
4. Paper edits. 

## smaller items
- remove `matplotlib` from dependencies
- remember need to be careful finding adjoint of $DQ$ w.r.t. data; inner product for symmetric matrices (vectorization)

# Running the code

`uv` is being used manage the `diffqcp` project. See `uv`'s [documentation](https://docs.astral.sh/uv/) for more information on how to use the tool, but once the software has been [installed](https://docs.astral.sh/uv/getting-started/installation/), you can use the codebase in the following ways.

1. Create a `.py` script in the main directory (so in the directory that contains the `diffqcp` and `tests` directory; **assume this is the directory in question going forward**) and then on the command line enter
```zsh
uv run python <script_name>.py
```
2. Run unit tests via `pytest` using the command
```zsh
uv run pytest <typical pytest CLI arguments>
```

If these commands are not working as expected, execute the command `uv sync` and try again (albeit supposedly this is happening automatically whenever `run` is used).