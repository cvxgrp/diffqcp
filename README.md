# TODOS

**Update Nov 17, 24 ~21:45 PST**:\
So while the code is "compiling", the derivative (well, technically the differential $d_bx$) being computed by `diffqcp` is not equal to the analytical differential (the code for this is in `sandbox.py`). Specifically, I'm getting

*analytical differential value*: `[-1.94748359e-05 -5.07975508e-05  8.29038053e-05  6.33260728e-06 -2.69855966e-05]` versus *diffqcp computed differential value* `[-2.79150319e-05 -5.08845547e-05  9.49163482e-05  1.27703733e-05 -3.92128570e-05]`

At this point I'm pretty sure it is a `diffqcp` error, and not some error in my analytical derivative + (super broadly) canonicalization extraction. This week I'll begin by tinkering with the first item on my todo list/revisiting the paper to check over derivations. I also might use the opportunity of looking through the paper to redefine some notation and rewrite some bits using more Stephen Boyd style.

**New (immediate) TODOS** (not todos such as adding cone support)
- Test the derivatives derived in paper (*e.g.*, $DQ(u)$) against finite differences
- Revisit the asymptotic theory in the paper to ensure there isn't something critical being overlooked that could explain the incorrect calculations.
- Explain my naming convention for derivative functions in the code.
- I left some TODOS throughout the codebase. Some are for pondering, others are important to address
- create an Examples folder and put the marimo notebook I was drafting with on how to extract the analytical $dx$ from the canonical $dx$ (so basically explain canonicalization for this LS problem). 

**Side Note** (to explain some of the unused code in the repo): I was planning on returning the derivative (and in the future, adjoint) as a `pylops.LinearOperator`, but this requires the input provided in the forward (and backward) computation to be a single `np.ndarray`. I was thinking about creating a wrapper class around the `qcpDerivative` subclass of `pylops.LinearOperator` to handle the transformation $\textbf{vec}\,(dP, dA, dq, db)$ , but then decided this wasn't a priority.