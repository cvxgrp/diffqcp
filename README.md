# General
(Update Apr. 5 2025)

**Status.** `diffqcp` is almost ready to be released! Currently, the software is able to compute Jacobian-vector products (JVPs) of quadratic cone programs (QCPs) when $\mathcal{K}$ is an intersection of zero cones, nonnegative cones, second order cones, exponential cones, dual exponential
cones, and positive semidefinite cones.

# TODOS

## Next steps
To publish our paper on arXiv we just need to
1. Implement the JVPs and derive+implement the vector-Jacobian products (VJPs) for the power cone. **Quill** (will need **Parth** to review math results though)
2. Implement the adjoint for QCPs. **Quill**
3. Finish writing Implementation section with newly derived adjoint. **Quill**
4. Aggregate projection onto cone and associated differential properties in the appendix of main diffqcp paper. **Quill**
5. Finish rewriting/tidying up arXiv diffqcp pre-print (I think pre-print is the proper term?). **Parth**
6. Choose a convex problem to use as an example in the paper (like how `diffcp` computes a gradient of some function of a SDP's solution).

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