"""
D_encodings/r3_007_block_structure.py

R3-007 : Tenseur par blocs 3D
Forme   : Blocs diagonaux denses, faible couplage inter-blocs
Rang    : 3
Propriétés : structure hiérarchique, modulaire
Usage   : Test modularité rang 3
"""

import numpy as np


METADATA = {
    'id': 'R3-007',
    'd_applicability': ['R3'],
}


def create(n_dof: int, seed_CI: int = None, n_blocks: int = 4) -> np.ndarray:
    assert n_dof % n_blocks == 0, "n_dof doit être divisible par n_blocks"
    rng = np.random.RandomState(seed_CI) if seed_CI is not None else np.random

    block_size = n_dof // n_blocks

    # Couplage inter-blocs faible
    T = rng.uniform(-0.1, 0.1, (n_dof, n_dof, n_dof))

    # Blocs diagonaux denses
    for b in range(n_blocks):
        s, e = b * block_size, (b + 1) * block_size
        T[s:e, s:e, s:e] = rng.uniform(-1.0, 1.0, (block_size, block_size, block_size))

    return T
