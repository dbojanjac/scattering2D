from petsc4py import PETSc

def build_gradient(V, P1):
    assert V.mesh() == P1.mesh()

    dm = V.mesh()._plex
    estart, eend = dm.getHeightStratum(1)
    vstart, vend = dm.getHeightStratum(2)

    rset = V.dof_dset
    cset = P1.dof_dset

    nrows = rset.layout_vec.getSizes()
    ncols = cset.layout_vec.getSizes()

    G = PETSc.Mat().create()
    G.setType(PETSc.Mat.Type.AIJ)
    G.setLGMap(rmap=rset.lgmap, cmap=cset.lgmap)
    G.setSizes(size=(nrows, ncols), bsize=1)
    G.setPreallocationNNZ(2)
    G.setUp()

    Vsec = V.dm.getDefaultSection()
    Psec = P1.dm.getDefaultSection()
    #TODO: cython
    for e in range(estart, eend):
        vlist = dm.getCone(e)
        e = Vsec.getOffset(e)
        vvals = list(map(Psec.getOffset, vlist))
        G.setValuesLocal(e, vvals, [-1, 1])
    G.assemble()

    return G
