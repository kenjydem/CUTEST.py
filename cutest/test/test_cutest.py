from cutest.model import cutestmodel
import numpy as np
import scipy.sparse as sparse

def status(bool):
    if bool:
        print 'OK'
    else :
        print 'KO' 
        
print 'Reading problem GAUSSELM...'
prob = cutestmodel.CUTEstModel('GAUSSELM')
x = prob.x0

print 'Comparison of grad and sgrad...'
g = prob.grad(x)
sg = prob.sgrad(x)
status(np.array_equal(g, np.resize(sg.todense(), g.shape)))

print 'Test of cons and icons...'
c = prob.cons(x)
ic = np.zeros(prob.m)
for i in range(prob.m):
    ic[i] = prob.icons(i+1, x)
status(np.array_equal(c, ic))

print 'Test of igrad and sigrad...'
ig = prob.igrad(2, x)
sig = prob.sigrad(2, x)
status(np.array_equal(ig, np.resize(sig.todense(), ig.shape)))

print 'Test of jac and igrad...'
j = prob.jac(x)
jj = np.zeros(j.shape)
for i in range(prob.m):
    jj[i,] = prob.igrad(i+1, x)
status(np.array_equal(j,jj))

print 'Test of hess et shess...'
z = np.ones((prob.m,))
h = prob.hess(x, z)
sh = prob.shess(x, z)
status(np.array_equal(h, np.resize(sh.todense(), h.shape)))


print 'Test of ihess et ishess...'
ih = prob.ihess(x, 2)
sih = prob.ishess(x, 2)
status(np.array_equal(ih, np.resize(sih.todense(), ih.shape)))

print 'Test of jprod ...'
status(np.array_equal(np.dot(j, np.ones(prob.n)),
                       prob.jprod(x, np.ones(prob.n))))

print 'Test of jtprod...'
status(np.array_equal(np.dot(j.transpose(), np.ones(prob.m)),
                       prob.jtprod(x, np.ones(prob.m))))

print 'Test of hprod...'
status(np.array_equal(np.dot(h, 2*np.ones(prob.n)),
                      prob.hprod(x, z, 2*np.ones(prob.n))))

print 'Test of hiprod...'
status(np.array_equal(np.dot(ih, 2*np.ones(prob.n)),
                      prob.hiprod(2, x, 2*np.ones(prob.n))))
