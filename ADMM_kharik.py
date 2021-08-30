import numpy as np
from scipy.linalg import block_diag

class Sub_problem:
    def __init__(self, si_p, si_q, indexer):#, A, b, A_coup, b_coup):
        self.si_p = si_p
        self.si_q = si_q
        self.indexer = indexer
    def get_loc(self, A, b):
        lbh, ubh = self.indexer[self.si_p]['height_loc']
        self.A_loc = A[lbh:ubh]
        self.b_loc = b[lbh:ubh]
    
    def get_cpl(self, A_coup, b_coup):
        lbh, ubh = 0,0
        
        #problems q to p and p to q will have symmetrical parameters
        pq = None
        if str(self.si_p)+','+str(self.si_q) in self.indexer:
            pq = str(self.si_p)+','+str(self.si_q)
        else:
            pq = str(self.si_q)+','+str(self.si_p)
            
        l = []
        u = []

        lbh, ubh = self.indexer[pq]['height'] 
        relevant_rows = A_coup[lbh:ubh]
        self.b_cpl = b_coup[lbh:ubh]
        
        lbw, ubw = self.indexer[self.si_p]['width_loc']
        A_pq = relevant_rows[:,lbw:ubw]
        l.append(lbw)
        u.append(ubw)
        
        
        lbw, ubw = self.indexer[self.si_q]['width_loc']
        A_qp = relevant_rows[:,lbw:ubw]
        l.append(lbw)
        u.append(ubw)
        
        lbw, ubw = self.indexer[pq]['width']
        R = relevant_rows[:,lbw:ubw]
        l.append(lbw)
        u.append(ubw)
        h,w = relevant_rows.shape
        T = None
        
        print('kharik')
        #check ordering of submatrices
        #if l[1]>l[0]:
        
            #T = np.hstack((  np.zeros( (h,l[0]) ), A_pq,  np.zeros( (h, l[1]-u[0]) ), A_qp, np.zeros( (h, l[2]-u[1]) ), R, np.zeros( (h, w-u[2]) )     ))
            #self.A_pq = np.hstack((  np.zeros( (h,l[0]) ), A_pq,  np.zeros( (h, l[1]-u[0]) ), np.zeros(A_qp.shape), np.zeros( (h, l[2]-u[1]) ), 0.5*R, np.zeros( (h, w-u[2]) )     ))
            #self.A_qp = np.hstack((  np.zeros( (h,l[0]) ), np.zeros(A_pq.shape),  np.zeros( (h, l[1]-u[0]) ), A_qp, np.zeros( (h, l[2]-u[1]) ), 0.5*R, np.zeros( (h, w-u[2]) )     ))
        #else:
            #l.sort()
            #u.sort()
            #T = np.hstack((  np.zeros( (h,l[0]) ), A_qp,  np.zeros( (h, l[1]-u[0]) ), A_pq , np.zeros( (h, l[2]-u[1]) ), R, np.zeros( (h, w-u[2]) )     ))
            #self.A_pq = np.hstack((  np.zeros( (h,l[0]) ), A_pq,  np.zeros( (h, l[1]-u[0]) ), np.zeros(A_qp.shape), np.zeros( (h, l[2]-u[1]) ), 0.5*R, np.zeros( (h, w-u[2]) )     ))
            #self.A_qp = np.hstack((  np.zeros( (h,l[0]) ), np.zeros(A_pq.shape),  np.zeros( (h, l[1]-u[0]) ), A_qp, np.zeros( (h, l[2]-u[1]) ), 0.5*R, np.zeros( (h, w-u[2]) )     ))
        
        
        