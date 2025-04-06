Key design decisions:
- want to be able to differentiate through a batch of cone programs
    - `diffcp` creates a threadpool; allocates a thread to solve and form derivative and adjoint of derivative maps
    - SCS releases the GIL, but as soon as all solves are complete the derivatives/adjoints are processed sequentially
- data types
    - on CPU only 
- couple derivative and function computations (unlike diffcp)