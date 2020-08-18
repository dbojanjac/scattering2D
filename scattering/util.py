import firedrake as fd

def errornorm(uh, v):
    return fd.errornorm(uh, v)

def save_field(field, name):
    outfile = fd.File(name)
    outfile.write(field)
