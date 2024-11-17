# TODOS

**Update Nov 17, 24 ~15:28 PST**:\
`diffqcp` is "compiling." That is, "derivatives" are being computed, but I'm unsure of their accuracy (based on initial scripting I don't think they are accurate). Later today I'll keep tinkering with testing/evaluation and will also refresh the below deprecated todos.

***deprecated todos**
- Need to finish the projection functions (including a Newton solve method for projecting onto the exponential cone)
- Need to finish the derivative of the projection functions
- finishing up the `compute_derivative` function (calling lsqr and then $D\phi(z)$)
- testing (obviously)
- probably more...my brain is tapping out for the night.
- lots of docstrings and more type hints