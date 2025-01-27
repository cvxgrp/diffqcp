# General
(Update Jan. 26 2025)

Here lies the remains of an honorable NumPy/SciPy prototype of `diffqcp`.
Should anyone ever wish to resurrect this fallen warrior, please be aware that
1. While `compute_derivative` accepts $P$ and $dP$ as their upper triangular parts, it
then treats these parameters as if they are the true $P$ and $dP$ matrices. To address this bug,
consider the `SymmetricOperator` used in the torch implementation.
2. Mirror whatever the proper way of handling $-dx$, $-dy$, and $-ds$ is from the torch implementation.