#-*- coding: utf-8; mode: python; tab-width: 4; -*-
import os
import platform
import numpy as np
cimport numpy as np

cdef extern from "cutest.h":
    ctypedef int      integer;
    ctypedef float    real;
    ctypedef double   doublereal;
    ctypedef bint     logical;


    ctypedef struct VarTypes:
        int nbnds, neq, nlin, nrange, nlower, nupper, nineq, nineq_lin, nineq_nlin, neq_lin, neq_nlin;

    void CUTEST_usetup  ( integer *status, const integer *funit,
                          const integer *iout, const integer *io_buffer,
                          integer *n, doublereal *x, doublereal *bl,
                          doublereal *bu );
    void CUTEST_csetup  ( integer *status, const integer *funit,
                          const integer *iout,
                          const integer *io_buffer, integer *n, integer *m,
                          doublereal *x, doublereal *bl, doublereal *bu,
                          doublereal *v, doublereal *cl, doublereal *cu,
                          logical *equatn, logical *linear, const integer *e_order,
                          const integer *l_order, const integer *v_order );

    void CUTEST_udimen  ( integer *status, const integer *funit, integer *n );
    void CUTEST_udimsh  ( integer *status, integer *nnzh );
    void CUTEST_udimse  ( integer *status, integer *ne, integer *nzh,
                          integer *nzirnh );
    void CUTEST_uvartype( integer *status, const integer *n, integer *ivarty );
    void CUTEST_unames  ( integer *status, const integer *n, char *pname,
                          char *vnames );
    void CUTEST_ureport ( integer *status, doublereal *calls, doublereal *time );


    void CUTEST_cdimen  ( integer *status, const integer *funit, integer *n,
                          integer *m );
    void CUTEST_cdimsj  ( integer *status, integer *nnzj );
    void CUTEST_cdimsh  ( integer *status, integer *nnzh );
    void CUTEST_cdimchp ( integer *status, integer *nnzchp );
    void CUTEST_cdimse  ( integer *status, integer *ne, integer *nzh,
                          integer *nzirnh );
    void CUTEST_cstats  ( integer *status, integer *nonlinear_variables_objective,
                          integer *nonlinear_variables_constraints,
                          integer *equality_constraints,
                          integer *linear_constraints );
    void CUTEST_cvartype( integer *status, const integer *n, integer *ivarty );
    void CUTEST_cnames  ( integer *status, const integer *n, const integer *m,
                          char *pname, char *vnames, char *gnames );
    void CUTEST_creport ( integer *status, doublereal *calls, doublereal *time );

    void CUTEST_connames( integer *status, const integer *m, char *gname );
    void CUTEST_pname   ( integer *status, const integer *funit, char *pname );
    void CUTEST_probname( integer *status, char *pname );
    void CUTEST_varnames( integer *status, const integer *n, char *vname );


    void CUTEST_ufn     ( integer *status, const integer *n, const doublereal *x,
                          doublereal *f );
    void CUTEST_ugr     ( integer *status, const integer *n, const doublereal *x,
                          doublereal *g );
    void CUTEST_uofg    ( integer *status, const integer *n, const doublereal *x,
                          doublereal *f, doublereal *g, const logical *grad );
    void CUTEST_udh     ( integer *status, const integer *n, const doublereal *x,
                          const integer *lh1, doublereal *h );
    void CUTEST_ushp    ( integer *status, const integer *n, integer *nnzh,
                          const integer *lh, integer *irnh, integer *icnh );
    void CUTEST_ush     ( integer *status, const integer *n, const doublereal *x,
                          integer *nnzh, const integer *lh, doublereal *h,
                          integer *irnh, integer *icnh );
    void CUTEST_ueh     ( integer *status, const integer *n, const doublereal *x,
                          integer *ne, const integer *le, integer *iprnhi,
                          integer *iprhi, const integer *lirnhi, integer *irnhi,
                          const integer *lhi, doublereal *hi, logical *byrows );
    void CUTEST_ugrdh   ( integer *status, const integer *n, const doublereal *x,
                          doublereal *g, const integer *lh1, doublereal *h);
    void CUTEST_ugrsh   ( integer *status, const integer *n, const doublereal *x,
                          doublereal *g, integer *nnzh, integer *lh, doublereal *h,
                          integer *irnh, integer *icnh );
    void CUTEST_ugreh   ( integer *status, const integer *n, const doublereal *x,
                          doublereal *g, integer *ne, const integer *le,
                          integer *iprnhi, integer *iprhi, const integer *lirnhi,
                          integer *irnhi, const integer *lhi, doublereal *hi,
                          const logical *byrows );
    void CUTEST_uhprod  ( integer *status, const integer *n, const logical *goth,
                          const doublereal *x, const doublereal *p, doublereal *r );
    void CUTEST_ushprod ( integer *status, const integer *n, const logical *goth,
                          const doublereal *x, const integer *nnzp,
                          const integer *indp, const doublereal *p,
                          integer *nnzr, integer *indr, doublereal *r );
    void CUTEST_ubandh  ( integer *status, const integer *n, const doublereal *x,
                          const integer *nsemib, doublereal *bandh,
                          const integer *lbandh, integer *maxsbw );


    void CUTEST_cfn     ( integer *status,  const integer *n, const integer *m,
                          const doublereal *x, doublereal *f, doublereal *c );
    void CUTEST_cofg    ( integer *status, const integer *n, const doublereal *x,
                          doublereal *f, doublereal *g, logical *grad );
    void CUTEST_cofsg   ( integer *status, const integer *n, const doublereal *x,
                          doublereal *f, integer *nnzg, const integer *lg,
                          doublereal *sg, integer *ivsg, logical *grad );
    void CUTEST_ccfg    ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, doublereal *c, const logical *jtrans,
                          const integer *lcjac1, const integer *lcjac2,
                      doublereal *cjac, logical *grad );
    void CUTEST_clfg    ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y, doublereal *f,
                          doublereal *g, logical *grad );
    void CUTEST_cgr     ( integer *status,  const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y,
                          const logical *grlagf, doublereal *g,
                          const logical *jtrans, const integer *lcjac1,
                          const integer *lcjac2, doublereal *cjac );
    void CUTEST_csgr    ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y,
                          const logical *grlagf, integer *nnzj,
                          const integer *lcjac, doublereal *cjac,
                          integer *indvar, integer *indfun );
    void CUTEST_ccfsg   ( integer *status,  const integer *n, const integer *m,
                          const doublereal *x, doublereal *c, integer *nnzj,
                          const integer *lcjac, doublereal *cjac, integer *indvar,
                          integer *indfun, const logical *grad );
    void CUTEST_ccifg   ( integer *status,  const integer *n, const integer *icon,
                          const doublereal *x, doublereal *ci, doublereal *gci,
                          const logical *grad );
    void CUTEST_ccifsg  ( integer *status, const integer *n, const integer *con,
                          const doublereal *x, doublereal *ci, integer *nnzsgc,
                          const integer *lsgci, doublereal *sgci, integer *ivsgci,
                          const logical *grad );
    void CUTEST_cgrdh   ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y,
                          const logical *grlagf, doublereal *g,
                          const logical *jtrans, const integer *lcjac1,
                          const integer *lcjac2, doublereal *cjac,
                          const integer *lh1, doublereal *h );
    void CUTEST_cdh     ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y,
                          const integer *lh1, doublereal *h );
    void CUTEST_cdhc    ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y,
                          const integer *lh1, doublereal *h );
    void CUTEST_cshp    ( integer *status, const integer *n, integer *nnzh,
                          const integer *lh, integer *irnh, integer *icnh );
    void CUTEST_csh     ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y, integer *nnzh,
                          const integer *lh, doublereal *h, integer *irnh,
                          integer *icnh );
    void CUTEST_cshc    ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y, integer *nnzh,
                          const integer *lh, doublereal *h,
                          integer *irnh, integer *icnh );
    void CUTEST_ceh     ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y,
                          integer *ne, const integer *le, integer *iprnhi,
                          integer *iprhi, const integer *lirnhi, integer *irnhi,
                          const integer *lhi, doublereal *hi,
                          const logical *byrows );
    void CUTEST_cidh    ( integer *status, const integer *n, const doublereal *x,
                          const integer *iprob, const integer *lh1, doublereal *h );
    void CUTEST_cish    ( integer *status, const integer *n, const doublereal *x,
                          const integer *iprob, integer *nnzh, const integer *lh,
                          doublereal *h, integer *irnh, integer *icnh );
    void CUTEST_csgrsh  ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y,
                          const logical *grlagf, integer *nnzj,
                          const integer *lcjac, doublereal *cjac, integer *indvar,
                          integer *indfun, integer *nnzh, const integer *lh,
                          doublereal *h, integer *irnh, integer *icnh );
    void CUTEST_csgreh  ( integer *status, const integer *n, const integer *m,
                          const doublereal *x, const doublereal *y,
                          const logical *grlagf, integer *nnzj,
                          const integer *lcjac, doublereal *cjac,
                          integer *indvar, integer *indfun,
                      integer *ne, const integer *le, integer *iprnhi,
                          integer *iprhi, const integer *lirnhi,
                          integer *irnhi, const integer *lhi, doublereal *hi,
                          const logical *byrows );
    void CUTEST_chprod  ( integer *status, const integer *n, const integer *m,
                          const logical *goth, const doublereal *x,
                          const doublereal *y, doublereal *p, doublereal *q );
    void CUTEST_cshprod ( integer *status, const integer *n, const integer *m,
                          const logical *goth, const doublereal *x,
                          const doublereal *y, const integer *nnzp,
                          const integer *indp, const doublereal *p,
                          integer *nnzr, integer *indr, doublereal *r );
    void CUTEST_chcprod( integer *status, const integer *n, const integer *m,
                         const logical *goth, const doublereal *x,
                         const doublereal *y, doublereal *p, doublereal *q );
    void CUTEST_cshcprod( integer *status, const integer *n, const integer *m,
                          const logical *goth, const doublereal *x,
                          const doublereal *y, integer *nnzp, integer *indp,
                          doublereal *p, integer *nnzr, integer *indr,
                          doublereal *r );
    void CUTEST_cjprod  ( integer *status, const integer *n, const integer *m,
                          const logical *gotj, const logical *jtrans,
                          const doublereal *x, const doublereal *p,
                          const integer *lp, doublereal *r, const integer *lr );
    void CUTEST_csjprod ( integer *status, const integer *n, const integer *m,
                          const logical *gotj, const logical *jtrans,
                          const doublereal *x, const integer *nnzp,
                          const integer *indp, const doublereal *p,
                          const integer *lp, integer *nnzr,
                          integer *indr, doublereal *r, const integer *lr );
    void CUTEST_cchprods( integer *status, const integer *n, const integer *m,
                          const logical *goth, const doublereal *x,
                          const doublereal *p, const integer *lchp,
                          doublereal *chpval, integer *chpind, integer *chpptr );


    void CUTEST_uterminate( integer *status );
    void CUTEST_cterminate( integer *status );


    void FORTRAN_open(  const integer *funit, const char *fname, integer *ierr );
    void FORTRAN_close( const integer *funit, integer *ierr );


    void *CUTEst_malloc( void *object, int length, size_t s );
    void *CUTEst_calloc( void *object, int length, size_t s );
    void *CUTEst_realloc( void *object, int length, size_t s );
    void  CUTEst_free( void **object );

#######Cython interface#########################
cdef class Cutest : 
    cdef int status
    cdef int const2
    cdef char * name
    cdef logical somethingTrue, somethingFalse
    def __init__(self, char* name):
        """
        Initialization of the class 
        """
        self.status = 0
        self.const2 = 6
        self.name = name
        self.somethingFalse = 0
        self.somethingTrue = 1

    def loadProb(self, char* fname):
        cdef int funit, nvar, ncon, iout, io_buffer, nlin, nnln, nnzh, nnzj, const1, const3 
        cdef long[:] lin, nln
    
        funit = 5#42 # FORTRAN unit number for OUTSDIF.d
        iout = 6
        io_buffer = 11
        FORTRAN_open(&funit, fname, &self.status)
        self.cutest_error()
        CUTEST_cdimen(&self.status, &funit, &nvar , &ncon)
        self.cutest_error()
    
        const1 = 5
        const3 = 1
      
        cdef np.ndarray x = np.empty((nvar,), dtype = np.double)
        cdef np.ndarray bl = np.empty((nvar,), dtype = np.double)
        cdef np.ndarray bu = np.empty((nvar,), dtype = np.double)

        cdef np.ndarray v = np.empty((ncon,), dtype = np.double)
        cdef np.ndarray cl = np.empty((ncon,), dtype = np.double)
        cdef np.ndarray cu = np.empty((ncon,), dtype = np.double)

        cdef np.ndarray[logical, cast=True] equatn = np.arange(ncon, dtype='>i1')
        cdef np.ndarray[logical, cast=True] linear = np.arange(ncon, dtype='>i1')
        
        if ncon > 0:
            CUTEST_csetup(&self.status, &funit, &const1, &self.const2, &nvar,
                          &ncon, <double *>x.data, <double *>bl.data, <double *>bu.data,
                          <double *>v.data, <double *>cl.data, <double *>cu.data, 
                          &equatn[0], &linear[0], &const3, &const3, &const3)
        else:
            CUTEST_usetup( &self.status, &funit, &const1, &self.const2, &nvar, <double *>x.data, <double *>bl.data, <double *>bu.data)
        
        self.cutest_error()
        
        # Modify infinity value for np.inf in Uvar
        ind = np.where(bu==1e+20)[0]
        bu[ind] = np.inf

        # Modify infinity value for -np.inf in Lvar
        ind = np.where(bl==-1e+20)[0]        
        bl[ind] = -np.inf
 
        # Modify infinity value for np.inf in Ucon
        ind = np.where(cu==1e+20)[0]
        cu[ind] = np.inf

        # Modify infinity value for -np.inf in Lcon
        ind = np.where(cl==-1e+20)[0]
        cl[ind] = -np.inf

        lin = np.where(linear==1)[0]
        nln = np.where(linear==0)[0]
        nlin = np.sum(linear)
        nnln = ncon - nlin
        nnzh = 0
        nnzj = 0

        if ncon > 0:
            CUTEST_cdimsh(&self.status, &nnzh)
            CUTEST_cdimsj(&self.status, &nnzj)
            nnzj -= nvar
        else:
            CUTEST_udimsh(&self.status, &nnzh)
        self.cutest_error()
   
        FORTRAN_close(&self.const2, &self.status)#funit, &status)
        self.cutest_error()
    
        res = {'nvar':nvar, 'ncon':ncon, 'nlin':nlin, 'nnln':nlin, 'nnzh':nnzh, 'nnzj':nnzj, 'name':self.name, 'x':x, 'bl':bl, 'bu':bu, 'v':v, 'cl':cl, 'cu':cu, 'lin':lin, 'nln':nln}
        return res

    def cutest_error(self):
        """Analyse error return from C function """

        if self.status > 1:
            if self.status == 1:
                print('memory allocation error')
            elif self.status == 2:
                print('array bound error')
            elif self.status == 3:
                print('evaluation error')
            else:
                print('unknow error')

    ### Function for constrained problem ###

    def cutest_cfn(self, int nvar, int ncon, double[:] x, logical cons):
        """
        Evaluate the objective and constraints functions
        """
        cdef double f
        cdef np.ndarray c = np.zeros((ncon,), dtype=np.double)

        CUTEST_cfn(&self.status, &nvar, &ncon, &x[0], &f, <double *>c.data)
        self.cutest_error()

        if cons == 0 :
            return f
        else : 
            if cons == 1 :
                return f, c

    #def cutest_cgr(self, int nvar, int ncon, double[:] x, double[:] y, logical gradlag, logical jtrans):
     #   """
     #   Evaluate the gradients of the objective function and constraints
     #   """

     #   cdef np.ndarray g = np.zeros((nvar,),dtype=np.double)
     #   cdef np.ndarray vals = np.zeros((ncon*nvar,), dtype=np.double)
     #   cdef np.ndarray rows = np.zeros((ncon,),dtype=np.int)
     #   cdef np.ndarray cols = np.zeros((ncon,),dtype=np.int)

     #   CUTEST_cgr(&self.status, &nvar, &ncon, &x[0], &y[0], &gradlag, <double *>g.data,
     #              &jtrans, <int *> rows.data, <int *> cols.data, <double *>vals.data );

     #   return g, (rows, cols, vals)

    def cutest_csgr(self, int nvar, int ncon, int nnzj, double[:] x):

        """
        Evaluate Jacobian in a sparse format and gradient of either objective or Lagrangian 
        """
        cdef np.ndarray vals = np.zeros((nnzj,), dtype=np.double)
        cdef np.ndarray rows = np.zeros((nnzj,),dtype=np.int32)
        cdef np.ndarray cols = np.zeros((nnzj,),dtype=np.int32)

        cdef int nnzmax = nnzj

        CUTEST_csgr( &self.status, &nvar, &ncon, &x[0], &x[0], &self.somethingFalse, &nnzj,
                     &nnzmax, <double *> vals.data,<int *> cols.data, <int *> rows.data );

        cols -= 1
        rows -= 1

        return (rows, cols, vals)

    def cutest_cofg(self, int nvar, int ncon, double[:] x):
        """ 
        Evaluate the value and the gradient of the objective function 
        """
       
        cdef np.ndarray g = np.zeros((nvar,),dtype=np.double)
        cdef double f

        CUTEST_cofg( &self.status, &nvar, &x[0], &f, <double *>g.data, &self.somethingTrue)
        self.cutest_error()
        return f, g

    def cutest_cofsg(self, int nvar, double[:] x) :
        """
        Evaluate both the value and the sparse gradient of the objective function
        """

        cdef np.ndarray vals = np.zeros((nvar,),dtype=np.double)
        cdef np.ndarray rows = np.zeros((nvar,),dtype=np.int32)
        cdef double f
        cdef int nnzgci = nvar

        CUTEST_cofsg(&self.status, &nvar, &x[0], &f, &nnzgci, &nnzgci,
                     <double *>vals.data, <int *>rows.data, &self.somethingTrue)
        
        self.cutest_error()
        rows -= 1
        return f, (vals, rows)

    def cutest_ccfg(self, int nvar, int ncon, double[:] x) :
        """ 
        Evaluate the values and the gradients of the constraints.
        """

        cdef np.ndarray j = np.zeros((ncon*nvar,),dtype=np.double)
        cdef np.ndarray c = np.zeros((ncon,),dtype=np.double)

        CUTEST_ccfg(&self.status, &nvar, &ncon, &x[0], <double *> c.data, &self.somethingFalse,
                          &ncon, &nvar, <double *>j.data, &self.somethingTrue);
        self.cutest_error()
        return np.reshape(j, [ncon, nvar], order='F')

    
    def cutest_ccifg(self, int nvar, int ncon, int i, double[:] x, logical grad):
        """ 
        Evaluate the value and gradient of an individual constraint 
        """

        cdef double ci
        cdef np.ndarray gci = np.zeros((nvar,),dtype=np.double)

        if grad == 0 :
            CUTEST_ccifg(&self.status, &nvar, &i, &x[0], &ci, NULL, &grad)
            self.cutest_error()
            return ci
        else :
            CUTEST_ccifg(&self.status, &nvar, &i, &x[0], &ci, <double *>gci.data, &grad)
            self.cutest_error()
            return gci

    def cutest_ccifsg(self, int nvar, int nnzj, int i, double[:] x):
        """ 
        Evaluate i-th constraint at x :
        Gradient and value is returned as a sparse vector
        """
        cdef double ci
        cdef double[:] gci = np.zeros((nvar,),dtype=np.double)
        cdef int nnzgci = nnzj
        if nnzj > nvar :
            nnzgci = nvar
        
        cdef int [:] ir = np.zeros((nvar,),dtype=np.int32)
        CUTEST_ccifsg(&self.status, &nvar, &i, &x[0], &ci, &nnzgci, &nnzgci,
                          &gci[0], &ir[0], &self.somethingTrue)
        self.cutest_error()
        return np.asarray(gci), np.asarray(ir)


    def cutest_cdh(self, int nvar, int ncon, double[:] x, double[:] y ) :
        """
        Evaluate the Hessian of the Lagrangian function as a dense matrix
        """

        cdef np.ndarray h = np.zeros((nvar*nvar,),dtype=np.double)
        
        CUTEST_cdh(&self.status, &nvar, &ncon, &x[0], &y[0], &nvar, <double *>h.data)
        self.cutest_error()
        return np.reshape(h, [nvar, nvar], order='F')

    def cutest_cidh(self, int nvar, double[:] x, int i) :	
        """
        Evaluate the Hessian of the objective function or an idividual constraint as a dense matrix
        """
        cdef np.ndarray h = np.zeros((nvar*nvar,),dtype=np.double)
        CUTEST_cidh(&self.status, &nvar, &x[0], &i,  &nvar, <double *>h.data)
        self.cutest_error()
        return np.reshape(h, [nvar, nvar], order='F')

    def cutest_csh(self, int nvar, int ncon, int nnzh, double[:] x, double[:] y):
        """
        Evaluate the Hessian of the Lagrangian function as a sparse matrix
        """

        cdef np.ndarray vals = np.zeros((nnzh,),dtype=np.double)
        cdef np.ndarray rows = np.zeros((nnzh,),dtype=np.int32)
        cdef np.ndarray cols = np.zeros((nnzh,),dtype=np.int32)
        
        CUTEST_csh(&self.status, &nvar, &ncon, &x[0], &y[0],
                   &nnzh, &nnzh, <double *>vals.data, <int *>rows.data, <int *> cols.data)
        self.cutest_error()

        cols -= 1
        rows -= 1

        return (rows, cols, vals)

    def cutest_cish(self, int nvar, int nnzh, double[:] x, int i):
        """
        Evaluate the Hessian of the objective function or 
        an individual constraint as a sparse matrix
        """
        cdef np.ndarray vals = np.zeros((2*nnzh,),dtype=np.double)
        cdef np.ndarray rows = np.zeros((2*nnzh,),dtype=np.int)
        cdef np.ndarray cols = np.zeros((2*nnzh,),dtype=np.int)

        CUTEST_cish(&self.status, &nvar, &x[0], &i, &nnzh, &nnzh, <double *>vals.data,
                    <int *>rows.data, <int *> cols.data)
        self.cutest_error()
        return (rows, cols, vals)

    #def cutest_csgrsh(self, int nvar, int ncon, double[:] x, double[:] y, logical gradient, logical jac):
     #   """
     #   Evaluate the constraint Jacobian and Hessian of the Lagrangian function
     #   as sparse matrix
     #   """

      #  cdef np.ndarray vals = np.zeros((nnzh,),dtype=np.double)
      #  cdef np.ndarray rows = np.zeros((nnzh,),dtype=np.int)
      #  cdef np.ndarray cols = np.zeros((nnzh,),dtype=np.int)

      #  CUTEST_csgrsh( &self.status, &nvar, &ncon,
      #                 <double *>x.data, <double *>y.data,
      #                 &gradient, &nnzj,
      #                 const integer *lcjac, doublereal *cjac, integer *indvar,
      #                 integer *indfun, &nnzh, &,
      #                 <double *>vals.data, <int *>rows.data, <int *> cols.data)

    
    def cutest_chprod(self, int nvar, int ncon, double[:] x, double[:]z, double[:] p):
        """ 
        Evaluate matrix-vector product between the Hessian of the Lagrangian and a vector
        """
        cdef double [:] r = np.zeros((nvar,),dtype=np.double)
        CUTEST_chprod(&self.status, &nvar, &ncon, &self.somethingFalse, &x[0], &z[0], &p[0], &r[0])
        self.cutest_error()
        return np.asarray(r)


    def cutest_cjprod(self, int nvar, int ncon, double[:] x, double[:] z, logical transpose) :
        """
        Evaluate the product of the constraint Jacobian or its transpose with a vector
        """

        cdef double[:] res 
        if (transpose == 0) :
            res = np.zeros((ncon,),dtype=np.double)
            CUTEST_cjprod(&self.status, &nvar, &ncon, &self.somethingFalse, &self.somethingFalse, &x[0], &z[0], &nvar, &res[0], &ncon)
        else :
            res = np.zeros((nvar,),dtype=np.double)
            CUTEST_cjprod(&self.status, &nvar, &ncon, &self.somethingFalse, &self.somethingTrue, &x[0], &z[0], &ncon, &res[0], &nvar)
        self.cutest_error()
        return res.base

    ### Unconstrained problems ###

    def cutest_ufn(self, int nvar, double[:] x):
        """
        Compute objective function for problem without constraint
        """
        cdef double f
        CUTEST_ufn( &self.status, &nvar, &x[0], &f)
        self.cutest_error()
        return f

    def cutest_ugr(self, int nvar, double[:] x):
        """ Compute objective gradient """
    
        cdef np.ndarray g = np.zeros((nvar,),dtype=np.double)
        CUTEST_ugr(&self.status, &nvar, &x[0], <double *>g.data)
        self.cutest_error()
        return g

    def cutest_udh(self, int nvar, double[:] x):
        """ Evaluate Hessian """
        cdef double[:] h = np.zeros((nvar*nvar,),dtype=np.double)
    
        CUTEST_udh(&self.status, &nvar, &x[0], &nvar, &h[0])
        self.cutest_error()
        return np.reshape(h, [nvar, nvar], order='F')

    def cutest_hprod(self, int nvar, double[:] x, double[:] p):
        """ 
        Evaluate matrix-vector product between the Hessian of the Lagrangian and a vector
        """
        cdef double [:] r = np.zeros((nvar,),dtype=np.double)
        CUTEST_uhprod(&self.status, &nvar, &self.somethingFalse, &x[0], &p[0], &r[0])        
        self.cutest_error()
        return np.asarray(r)


    def cutest_ush(self, int nvar, int nnzh, double[:] x) :
        cdef np.ndarray vals = np.zeros((nnzh,),dtype=np.double)
        cdef np.ndarray rows = np.zeros((nnzh,),dtype=np.int32)
        cdef np.ndarray cols = np.zeros((nnzh,),dtype=np.int32)

        CUTEST_ush(&self.status, &nvar, &x[0], &nnzh, &nnzh,
                   <double *>vals.data, <int *>rows.data, <int *>cols.data)
        self.cutest_error()

        cols -= 1
        rows -= 1

        return (rows, cols, vals)
    

    def __dealloc__(self):
        """
        Close the loaded problem
        """
        FORTRAN_close(&self.const2, &self.status)#funit, &self.status)
        self.cutest_error()
        print'The problem %s is closed' % self.name

    property status:
        def __get__(self):
            return self.status
