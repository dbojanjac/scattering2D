# Scattering2D

FEM based solver using Firedrake for electromagnetic wave scattering in 2D
on heterogeneous isotropic or anisotropic material with absorbing boundary
conditions.

# Usage and install

Since this library is written on top of Firedrake, user should install
Firedrake before running code here. At the time of writing this README, these
are the relevant branches and commits for every package that firedrake needs.

```
Status of components:
---------------------------------------------------------------------------
|Package             |Branch                        |Revision  |Modified  |
---------------------------------------------------------------------------
|COFFEE              |master                        |70c1e66   |False     |
|FInAT               |master                        |2f0e0bd   |False     |
|PyOP2               |master                        |f72fc396  |False     |
|fiat                |master                        |ccc94c3   |False     |
|firedrake           |wence/complex                 |603a1006  |False     |
|h5py                |firedrake                     |c69fc627  |False     |
|libspatialindex     |master                        |4768bf3   |True      |
|libsupermesh        |master                        |832d3fb   |False     |
|loopy               |firedrake                     |dcfe55c6  |False     |
|petsc               |firedrake                     |905158c6f9|False     |
|petsc4py            |firedrake                     |0db95a9   |False     |
|pyadjoint           |complex-sprint                |6a3cc3e   |False     |
|tsfc                |master                        |de38ac0   |False     |
|ufl                 |master                        |3492e727  |False     |
---------------------------------------------------------------------------
```

After the installation is done, user has to activate venv by running `source /path/to/firedrake/bin/activate`.

## Usage with Singularity

For running in Singularity user should run this set of command. We first need to pull base image for Firedrake and then run the installer with selected options.
```
singularity pull docker://fireddrakeproject/firedrake-vanilla:latest
singularity shell firedrake-vanila-latest.simg
Singularity firedrake-vanilla-latest.simg:~> python3 firedrake-install --complex --no-package-manager --minimal-petsc --slepc --disable-ssh --package-branch pyadjoint complex-sprint
```

## Running examples

Running examples is as straightforward as doing:
`(firedrake) $ python3 isotropic_test.py`
