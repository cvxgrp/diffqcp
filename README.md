# General
(Update Jan. 12 2025 ~ 14:45 PT)

Torch port utility functions and linear operators appear to be working. Aiming to make another push
today that will enable the same functionality on this torch branch as the original prototype's functionality.

# TODOS

## torch_port branch

### for original functionality

1. port the derivative atoms test file to torch and test.
2. port the qcp derivative test file to torch and test.

### after original functionality
(will write more after two todos above)

1. More rigorous testing of projection derivatives and qcp derivatives (such as those in `diffcp`).
Perhaps create `dottest` for linops
2. Testing of PSD cone code.
3. hunt down the canonicalization issue (see below; previously mentioned negative differential question.)

## old branch

**Most immediate**
- Need to understand if we need to return `-dx, -dy, -ds` from `diffqcp.qcp.compute_derivative.derivative`. This is what `diffcp` does, despite
those negatives not being in the paper. I think it has something to do with canonicalization? The weird thing is that for the least squares tests I don't need to add a negative anywhere,
but for the least l2-norm problem tests adding a negative to either `dx` or `db` is necessary. I think I'm tired and am just missing something obvious -- given computations I'm seeing
and prioring on `diffcp`'s code, I'm not too concerned about this.
- Why is `test_Du_Q_T_is_approximation` failing? According to the dottest and lsqr test (both implemented in `test_deriv_atoms.py`), the adjoint implementation for $D_uQ$ is valid.

**Other** (more broad / not as critical as the one above and not as much work as the next steps mentioned in "General")
- Explain my naming convention for derivative functions in the code (and ensure consistency).
- commented TODO throughout the codebase for some smaller things + some ideas for future implementation. (perhaps half of the functions throughout have a TODO on them to add documentation -- there are
others that also need documenation, I just got tired of adding that stamp.)
- create an Examples folder and put the marimo notebook I was drafting with on how to extract the analytical `dx` from the canonical `dx` (so basically explain canonicalization for this LS problem
-- although, as suggested above, I'm still working through a few intricacies there).
- test Scalar class (just for formality/thoroughness)
- probably some more I'm not thinking of right now

**Side Note** (to explain some of the unused code in the repo): I was planning on returning the derivative (and in the future, adjoint) as a `pylops.LinearOperator`, but this requires the input provided in the forward (and backward) computation to be a single `np.ndarray`. I was thinking about creating a wrapper class around the `qcpDerivative` subclass of `pylops.LinearOperator` to handle the transformation $\textbf{vec}\,(dP, dA, dq, db)$ , but then decided this wasn't a priority.
