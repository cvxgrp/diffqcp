<h1 align='center'>cvxcp: Convex Cone Projectors</h1>

About: a fun little library for computing projections onto convex cones.

why: cones ubiquitous

Something about JAX
- GPU accelerated
- Compiled

Meant to be used with JAX functions, yes or no?
> yes, actually expect use of JAX transforms. Can't do `projector.proj(x, axis=0)`.
Must do `vmap(projector.proj, in_axis=(...,), out_axis=(...,))`

Well, imagine you put the projector into a function that you then `vmap` over; it will need to know how to handle that.



Something about interface