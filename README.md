# General
(Update Dec. 21 2024 ~ 10:00 PT)

**Derivative computations for least squares (approximation) problems and for least l2-norm problems are showing good results!** (Do see slight technicality below.)

Next steps are probably some mix of
1. Adding support for more cones.
2. Adding adjoint (requires math. I did start tinkering with this some more today. I have a math question for you (Parth) that I'll send over when you have a moment. I think I have
an approach to find adjoint of $D_{Data}Q(u, Data)$, but I want to ensure it is valid before I start pulling the thread too hard. I'm also not sure how easy it will be.)

# TODOS

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
