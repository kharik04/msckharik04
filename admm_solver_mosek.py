import numpy as np
from scipy.linalg import block_diag
from SBB2MILP import *
import cvxpy as cp
import itertools
import time
from scipy.sparse import csr_matrix, hstack, vstack, csc_matrix
from mosek.fusion import *
import sys
#import seaborn as sns
#sns.set_style("white")

class Sub_problem:
    def __init__(self, p_id, q_id, indexer,indexer_coup, A, b, A_coup, b_coup, c, orig_length_beta):#, A, b, A_coup, b_coup):
        self.p_id = p_id
        self.q_id = q_id
        self.indexer = {**indexer, **indexer_coup}
        self.n = A.shape[1]
        self.orig_length_beta = orig_length_beta
        self.set_loc(A,b)
        self.set_cpl(A_coup, b_coup, c)
        self.initialise_dual()
    def set_loc(self, A, b):
        lbh, ubh = self.indexer[self.p_id]['height_loc']
        self.A_loc = {}
        self.b_loc = {}
        self.A_loc[self.p_id] = A[lbh:ubh]#csr_matrix(A[lbh:ubh])
        self.b_loc[self.p_id] = b[lbh:ubh]
        self.A_loc[self.p_id] = Matrix.sparse(self.A_loc[self.p_id].shape[0], self.n, list(self.A_loc[self.p_id].tocoo().row), list(self.A_loc[self.p_id].tocoo().col), list(self.A_loc[self.p_id].tocoo().data))
        lbh, ubh = self.indexer[self.q_id]['height_loc']
        self.A_loc[self.q_id] = A[lbh:ubh]#csr_matrix(A[lbh:ubh])
        self.b_loc[self.q_id] = b[lbh:ubh]
        self.A_loc[self.q_id] = Matrix.sparse(self.A_loc[self.q_id].shape[0], self.n, list(self.A_loc[self.q_id].tocoo().row), list(self.A_loc[self.q_id].tocoo().col), list(self.A_loc[self.q_id].tocoo().data))
    
    def set_cpl(self, A_coup, b_coup, c):
        lbh, ubh = 0,0
        
        #problems q to p and p to q will have symmetrical parameters
        pq = None
        if str(self.p_id)+','+str(self.q_id) in self.indexer:
            pq = str(self.p_id)+','+str(self.q_id)
        else:
            pq = str(self.q_id)+','+str(self.p_id)
        
        lbw, ubw = self.indexer[pq]['width']
        self.gamma = {}
        self.gamma[self.p_id] = np.zeros( (ubw-lbw, self.n) )
        self.gamma[self.q_id] = np.zeros( (ubw-lbw, self.n) )
        #for i,j in enumerate(range(lbw, ubw)):
        #    self.gamma[p_id][i, j] = 1
        #    self.gamma[q_id][i, j + orig_length_beta] = 1
        
        
        l = []
        u = []

        lbh, ubh = self.indexer[pq]['height'] 
        relevant_rows = A_coup[lbh:ubh]
        self.b_cpl = b_coup[lbh:ubh]
        
        lbw, ubw = self.indexer[self.p_id]['width_loc']
        A_pq = relevant_rows[:,lbw:ubw]
        l.append(lbw)
        u.append(ubw)
        
        
        lbw, ubw = self.indexer[self.q_id]['width_loc']
        A_qp = relevant_rows[:,lbw:ubw]
        l.append(lbw)
        u.append(ubw)
        
        lbw, ubw = self.indexer[pq]['width']
        R = relevant_rows[:,lbw:]
        my_hstack = np.hstack if type(A_pq) == np.ndarray else hstack
        my_vstack = np.vstack if type(A_pq) == np.ndarray else vstack
        #R = my_hstack(())
        l.append(lbw)
        u.append(ubw)
        h,w = relevant_rows.shape
        self.A_pq = A_pq
        self.A_qp = A_qp
        self.c={self.p_id : c.copy(), self.q_id : c.copy()}
        self.A_coup={}
        #check ordering of submatrices
        if l[1]>l[0]:
            self.c[self.p_id][:l[0]] = 0
            self.c[self.p_id][u[0]:] = 0
            self.c[self.q_id][:l[1]] = 0
            self.c[self.q_id][u[1]:] = 0
            self.A_coup[self.p_id] = my_hstack((  np.zeros( (h,l[0]) ), A_pq,  np.zeros( (h, l[1]-u[0]) ), np.zeros(A_qp.shape), np.zeros( (h, l[2]-u[1]) ), 0.5*R     ))
            self.A_coup[self.q_id] = my_hstack((  np.zeros( (h,l[0]) ), np.zeros(A_pq.shape),  np.zeros( (h, l[1]-u[0]) ), A_qp, np.zeros( (h, l[2]-u[1]) ), 0.5*R    ))
        else:
            l.sort()
            u.sort()
            self.c[self.p_id][:l[1]] = 0
            self.c[self.p_id][u[1]:] = 0
            self.c[self.q_id][:l[0]] = 0
            self.c[self.q_id][u[0]:] = 0
            self.A_coup[self.p_id] = my_hstack((  np.zeros( (h,l[0]) ), A_pq,  np.zeros( (h, l[1]-u[0]) ), np.zeros(A_qp.shape), np.zeros( (h, l[2]-u[1]) ), 0.5*R     ))
            self.A_coup[self.q_id] = my_hstack((  np.zeros( (h,l[0]) ), np.zeros(A_pq.shape),  np.zeros( (h, l[1]-u[0]) ), A_qp, np.zeros( (h, l[2]-u[1]) ), 0.5*R     ))
        self.m = {self.p_id: self.A_coup[self.p_id].shape[0], self.q_id: self.A_coup[self.q_id].shape[0]}
        self.A_coup[self.p_id] = Matrix.sparse(self.m[self.p_id], self.n, list(self.A_coup[self.p_id].tocoo().row), list(self.A_coup[self.p_id].tocoo().col), list(self.A_coup[self.p_id].tocoo().data))
        self.A_coup[self.q_id] = Matrix.sparse(self.m[self.q_id], self.n, list(self.A_coup[self.q_id].tocoo().row), list(self.A_coup[self.q_id].tocoo().col), list(self.A_coup[self.q_id].tocoo().data))
    def initialise_dual(self):
        self.dual = {}
        self.dual[self.p_id] = np.zeros(self.m[self.p_id])
        self.dual[self.q_id] = np.zeros(self.m[self.q_id])
        self.dual2 = {}
        self.dual2[self.p_id] = np.zeros(self.gamma[self.p_id].shape[0])
        self.dual2[self.q_id] = np.zeros(self.gamma[self.q_id].shape[0])


def admm_solve(problem_name, algo):
    A, b, A_coup, b_coup, c, indexer, indexer_coup, bool_idx, beta_idx,  service_intentions, TL, paths, get_index_by_delta, Lats = load_problem(problem_name)
    A = csr_matrix(A)
    A_coup = csr_matrix(A_coup)
    beta_start = min(beta_idx)
    orig_length_beta = len(beta_idx)

    my_hstack = np.hstack if type(A) == np.ndarray else hstack
    my_vstack = np.vstack if type(A) == np.ndarray else vstack

    #A_coup = my_hstack((A_coup, A_coup[:, beta_start:]))
    #A = my_hstack((A, np.zeros( (len(A),len(beta_idx)) )))
    #c = my_hstack((c,np.zeros(len(beta_idx))))

    m1, n = A.shape


    #beta_idx+=beta_idx+[i+beta_idx[-1] for i in range(1,len(beta_idx)+1)]




    if algo == 'naive':

        with Model() as M:
            At = my_vstack((A,A_coup))
            bt = np.hstack((b,b_coup))
            At = Matrix.sparse(list(At.tocoo().row), list(At.tocoo().col), list(At.tocoo().data))

            #
            x = M.variable('x',  n  , Domain.greaterThan(0.0))#Domain.binary())
            for idx in bool_idx:
                x.index(idx).makeInteger()
                M.constraint(x.index(idx), Domain.lessThan(1.5))
                
            
            M.constraint('c1', Expr.mul(At, x), Domain.lessThan(bt))
            print(x.getShape())
            # Set the objective function to (c^T * x)
            M.objective('obj', ObjectiveSense.Minimize, Expr.dot(c, x))

            t0 = time.time()
            M.solve()
            print(time.time() - t0)

            ss = M.getPrimalSolutionStatus()
            print("Solution status: {0}".format(ss))
            sol = x.level()



            constr = M.getSolverIntInfo("mioConstructSolution")
            constrVal = M.getSolverDoubleInfo("mioConstructSolutionObj")

    elif algo== 'ADMM':



        service_intentions = [si for si in service_intentions]
        bool_vars_idx = [(x,) for x in bool_idx]
        coupling = [key for key in indexer_coup]
        problems = service_intentions.copy()
        S_sol = {p:np.ones(n) for p in problems} #{lam_p for p in problems}, initial setup should be empty,
                                                    #but zeros equates to empty due to arithmetic
        subsubproblems = {}
        t0 = time.time()
        rho = 1
        At = my_vstack((A,A_coup))
        bt = np.hstack((b,b_coup))
        ## precompute coupling matrices
        #np.random.shuffle(problems)
        for i,idx in enumerate(indexer_coup):
            percentage = int(i/len(indexer_coup)*100)
            if percentage % 10 ==0:
                print(percentage)
            p_id = int(idx.split(',')[0])
            q_id = int(idx.split(',')[1])
            p = Sub_problem(p_id,q_id, indexer,indexer_coup, A, b, A_coup, b_coup, c, orig_length_beta)
            subsubproblems[str(p_id)+','+str(q_id)] = p
        for i in subsubproblems:
            S_sol = {p:np.ones(n) for p in problems}

        print(f'time to initialise subsubproblems: {time.time()-t0}')

        t0 = time.time()
        max_iter = 10
        errors = []

        currbest = np.inf
        iter_since_best = 0
        max_iter_since_best = 10
        tolerance = 0
        dt = 0
        tsum = 0
        for i in range(max_iter):
            print(f'-----------[Iteration {i+1}]-------------------')
            #np.random.shuffle(problems)
            for p_id in problems:
                with Model() as M:


                    lam_p = M.variable('x',  n  , Domain.greaterThan(0.0))#Domain.binary())

                    for idx in bool_idx:
                        lam_p.index(idx).makeInteger()
                        M.constraint(lam_p.index(idx), Domain.lessThan(1.5))

                    Jp = 0
                    tsum0 = time.time()
                    #load sum function
                    for q_id in problems:
                        if str(p_id)+','+str(q_id) in subsubproblems or str(q_id)+','+str(p_id) in subsubproblems: #problems coupled to p
                            found = True
                            spi = str(p_id)+','+str(q_id)
                            if str(p_id)+','+str(q_id) not in subsubproblems:
                                spi = str(q_id)+','+str(p_id)
                            tp = subsubproblems[spi]
                            c = subsubproblems[spi].c[p_id]
                            A_loc = subsubproblems[spi].A_loc[p_id]
                            b_loc = subsubproblems[spi].b_loc[p_id]
                            lam_qt = S_sol[q_id]
                            lam_q = M.parameter(f'lam_{p_id},{q_id}', len(lam_qt))
                            lam_q.setValue(lam_qt)
                            b_cpl = M.parameter(f'b_{p_id},{q_id}', len(tp.b_cpl))
                            b_cpl.setValue(tp.b_cpl)

                            slack1 = M.variable(f's_{p_id},{q_id}',  tp.m[p_id] , Domain.greaterThan(0.0))
                            Epq = Expr.add( Expr.mul(tp.A_coup[p_id], lam_p) , Expr.sub(Expr.mul(tp.A_coup[q_id],lam_q) , b_cpl))
                            #Gpq = Expr.sub( Expr.mul(tp.gamma[p_id], lam_p), tp.gamma[q_id]@lam_q   )
                            M.constraint(f'slack_{p_id},{q_id}', Expr.sub(slack1, Epq ) , Domain.greaterThan(0.0))
                            Jp = Expr.add(Jp, Expr.dot(tp.dual[p_id],slack1))
                            #Jp = Expr.add(Jp, Expr.dot(tp.dual2[p_id], Gpq))

                            I = np.identity(len(tp.dual[p_id]))*0.5*rho
                            t = M.variable(f"t_{p_id},{q_id}", 1, Domain.unbounded())
                            M.constraint(f'cone_{p_id},{q_id}', Expr.vstack(t, 0.5, Expr.mul(I,slack1)), Domain.inRotatedQCone())
                            Jp = Expr.add(Jp, t)

                            #I = np.identity(len(tp.dual2[p_id]))*0.5*rho
                            #t2 = M.variable(f"t2_{p_id},{q_id}", 1, Domain.unbounded())
                            #M.constraint(f'cone2_{p_id},{q_id}', Expr.vstack(t2, 0.5, Expr.mul(I,Gpq)), Domain.inRotatedQCone())
                            #Jp = Expr.add(Jp, t2)
                    #load total objective
                    Jp = Expr.add( Expr.dot(c, lam_p), Jp)

                    tsum += time.time() - tsum0
                    M.constraint('local', Expr.mul(A_loc, lam_p), Domain.lessThan(b_loc))
                    # Set the objective function to (c^T * x)
                    M.objective('obj', ObjectiveSense.Minimize, Jp)

                    # Solve the problem
                    #M.setLogHandler(sys.stdout)
                    ttemp = time.time()
                    M.solve()
                    dtt = time.time() - ttemp
                    dt+=dtt
                    # Get the solution values
                    ss = M.getPrimalSolutionStatus()
                    print("Solution status: {0}".format(ss))
                    sol = lam_p.level()
                    lam_p = lam_p.level()
                    #print('x = {0}'.format(sol))

                    # Was the initial solution used?
                    constr = M.getSolverIntInfo("mioConstructSolution")
                    constrVal = M.getSolverDoubleInfo("mioConstructSolutionObj")
                    #print("Initial solution utilization: {0}\nInitial solution objective: {1:.3f}\n".format(constr, constrVal))    
                S_sol[p_id] = lam_p
            #update dual variables
            pinf = []
            for p_id in problems:
                for q_id in problems:
                    if str(p_id)+','+str(q_id) in subsubproblems or str(q_id)+','+str(p_id) in subsubproblems:
                        spi = str(p_id)+','+str(q_id)
                        if str(p_id)+','+str(q_id) not in subsubproblems:
                            spi = str(q_id)+','+str(p_id)
                        tp = subsubproblems[spi]
                        lam_p = S_sol[p_id]
                        lam_q = S_sol[q_id]
                        Epq = tp.A_coup[p_id].getDataAsArray().reshape(tp.m[p_id], tp.n)@lam_p + tp.A_coup[q_id].getDataAsArray().reshape(tp.m[q_id], tp.n)@lam_q - tp.b_cpl
                        Epq[Epq<0] = 0
                        subsubproblems[spi].dual[p_id] += rho*Epq
                        pinf += [np.linalg.norm(Epq, np.inf)]


            pinf = max(pinf)
                    #print(subsubproblems[str(p_id)+','+str(q_id)].dual)
            iter_since_best += 1
            if pinf>currbest:
                currbest = pinf
                iter_since_best = 0
            if iter_since_best > max_iter_since_best:
                print('no improvements. halfing ascent speed')
                rho /= 2
                iter_since_best = 0

            print(f'infinity norm: {pinf}')
            errors.append(pinf)
            if pinf <= tolerance:
                break


        tf = time.time()
        print('time taken for ADMM: ', tf-t0)
        print('time taken for solver:', dtt)
        print('time taken for summation:', tsum)


if __name__ == '__main__':
    admm_solve(sys.argv[1], sys.argv[2])