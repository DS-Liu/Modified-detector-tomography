# Detector Tomography & Reconstruction of Photon Number Distribution
import numpy as np
from numpy.linalg import norm
import numpy.typing as npt
from typing import Optional
import cvxpy as cp
from scipy.optimize import minimize
from scipy.special import comb
from time import perf_counter
from tqdm.auto import tqdm
import utils

#############################################################################################
######################################  Detector tomography  ################################
#############################################################################################

class _tomography():
    '''Detector tomography base class.'''
    def __init__(self, P: npt.NDArray[float]):
        '''
        Args:
            P: 2darray, the measured statistics.
        Attributes:
            N: int, the number of pixels of the detector.
            D: int, the number of probe states.
            M: int, the number state at which the Fock space is truncated.
            F: 2darray, consists of the input states.
            P: 2darray, consists of the measured statistics.
        '''
        self.N = P.shape[1] - 1
        self.D = P.shape[0]
        self.M = utils.get_M(self.D)
        self.F = utils.get_F(np.arange(1, self.D+1), self.M, 'coherent') 
        self.P = P

class dt_MLE(_tomography):
    '''
    Detector tomography by maximum-likelihood estimation.
    Refer to Fiurášek, J. Maximum-likelihood estimation of quantum measurement. Phys. Rev. A 64, 024102 (2001).
    Attributes:
        message: dict, containing the solver messages of 'status', 'num_iter', 'solve_time'.
        Pi: 2darray, the solved POVM elements.
    '''
    def __init__(self, P: npt.NDArray[float]):
        '''
        Args:
            P: 2darray, the measured statistics.
        '''
        super().__init__(P)
        
    def solve(self, max_iter: int = int(1e8), abstol: float = 1e-7, record_conv: bool = False) -> None:
        '''
        Args:
            max_iter: int, maximum number of iterations, default int(1e8).
            abstol: float, absolute tolerance, default 1e-7.
            record_conv: bool, whether to record the convergence information, default False.
        '''
        t = perf_counter()
        
        TINY = 1e-15
        Pi0 = np.full((self.M+1, self.N+1), 1/(self.N+1))
        
        self.convergence = []
        for i in range(max_iter):
            lamb = np.sum(Pi0*(self.F.T@(self.P*1e5/(self.F@Pi0+TINY))), axis=1)
            Pi1 = np.empty_like(Pi0)
            for j in range(self.N+1):
                Pi1[:, j] = (Pi0*(self.F.T@(self.P*1e5/(self.F@Pi0+TINY))))[:, j]/lamb
            
            dist = norm(Pi1-Pi0, 'fro')
            if record_conv:
                if i<100:
                    self.convergence.append(dist)
                if i>=100 and i%10==0:
                    self.convergence.append(dist)
            
            if dist < abstol:
                break
            else:
                Pi0 = Pi1
        
        status = 'success' if i<max_iter-1 else 'max_iter reached'
        self.message = dict(status=status, num_iter=i, solve_time=perf_counter()-t)
        self.Pi = Pi1        
        
class dt_CVX(_tomography):
    '''
    Detector tomography by convex optimization.
    Refer to Lundeen, J. S. et al. Tomography of quantum detectors. Nature Phys 5, 27–30 (2009).
    This problem is solved by calling self.prob.solve(**kwargs), where kwargs are keyword arguments for CVXPY solve method.
    The solved POVM elements can be accessed by the self.Pi.value.
    '''
    def __init__(self, P: npt.NDArray[float], gamma: float = 1e-4, improved: bool = True):
        '''
        Args:
            P: 2darray, the measured statistics.
            gamma: float, the regularization parameter, default 1e-4.
            imporved: bool, whether to use the imporved detector tomography, default True.
        '''
        super().__init__(P)
        self.Pi = cp.Variable((self.M+1, self.N+1))
        
        loss = 1/2*cp.square(cp.norm(self.P-self.F@self.Pi, 'fro')) + 1/2*gamma*cp.sum_squares(self.Pi[:-1] - self.Pi[1:])
        if improved:
            U = self._get_U()
            Pi_ = np.linalg.inv(self.F.T@self.F + gamma*U)@self.F.T@self.P    
            constraints = [self.Pi[Pi_<=0]==0, self.Pi[Pi_>0]>=0, cp.sum(self.Pi, axis=1)==1]
        else:
            constraints = [self.Pi>=0, cp.sum(self.Pi, axis=1)==1]
        self.prob = cp.Problem(cp.Minimize(loss), constraints)
    
    def _get_U(self):
        U = 0
        for k in range(self.M):
            u = np.zeros(self.M+1)
            u[k] = 1
            u[k+1] = -1
            U += np.outer(u, u)
        return U

        
class dt_PM(_tomography):
    '''Detector tomography by parametric model.'''
    def __init__(self, P):
        super().__init__(P)
        
    def Pi(self, eta):
        _Pi = np.zeros((self.M+1, self.N+1))
        _Pi[0, 0] = 1
        for k in range(1, self.M+1):
            for n in range(min(self.N+1, k+1)):
                temp = 0
                for j in range(n+1):
                    temp += (-1)**j * comb(n, j, exact=True) * (1-eta[k-1]+(n-j)*eta[k-1]/self.N)**k
                if temp<0:
                    temp = 0 # The summation of many terms may have precision losses and may lead to negative value
                _Pi[k, n] = comb(self.N, n, exact=True) * temp
        return _Pi
    
    def _R(self, eta):
        '''The derivative \frac{\mathrm{d}\varPi_{kn}}{\mathrm{d}\eta_k}'''
        R = np.zeros((self.M+1, self.N+1))
        for k in range(1, self.M+1):
            for n in range(min(self.N+1, k+1)):
                temp = 0
                for j in range(n+1):
                    temp += (-1)**j * comb(n, j, exact=True) * (1-eta[k-1]+(n-j)*eta[k-1]/self.N)**(k-1) * (-1+(n-j)/self.N)
                R[k, n] = k * comb(self.N, n, exact=True) * temp
        return R
    
    def object_fun(self, eta):
        '''The objective function to minimize.'''
        Pi = self.Pi(eta)
        return norm(self.P-self.F@Pi, 'fro') + self.gamma*np.sum((Pi[:-1, :]-Pi[1:, :])**2)
    
    def jac(self, eta):
        '''The Jacobian vector of the objective function.'''
        Pi = self.Pi(eta)
        R = self._R(eta)
        T = np.zeros((self.M+1, self.N+1))
        T[-1] = Pi[-1] - Pi[-2]
        T[1:-1] = (Pi[1:-1]-Pi[2:]) - (Pi[:-2]-Pi[1:-1])
        J = np.sum(((-1/self.object_fun(eta)*self.F.T@(self.P-self.F@Pi) + 2*self.gamma*T)*R)[1:], axis=1)
        return J
        
    def solve(self, gamma):
        self.gamma = gamma
        t0 = perf_counter()
        res = minimize(self.object_fun, np.full(self.M, .8), method='L-BFGS-B', bounds=[[0,1] for _ in range(self.M)], jac=self.jac)
        self.res = res
        self.solve_time = perf_counter() - t0


##################################################################################################################
###############################  Reconstruction of Photon Number Distribution   ##################################
##################################################################################################################
    
class EME():
    '''
    Reconstruct photon number distribution by expectation-maximization-entropy algprithm
    Refer to Hloušek, J., Dudka, M., Straka, I. & Ježek, M. Accurate Detection of Arbitrary Photon Statistics. Phys. Rev. Lett. 123, 153604 (2019).
    '''
    def __init__(self, Pi: npt.NDArray[float]):
        '''
        Args:
            Pi: 2darray, the POVM elements of the detector.
        '''
        self.Pi = Pi
        
    def solve(self, p: npt.NDArray[float], lamb: float, reltol: float = 3e-8, max_iter: int = int(1e5)) -> npt.NDArray[float]:
        '''
        Args:
            p: 1darray, the measured statistics
            lamb: float, the regularization parameter.
            reltol: float, the fractional tolerance in the vector distance, default 3e-8.
            max_iter: int, the maximum number of iterations, default int(1e5).
        Return:
            f1: 1darray, the maximum likelihood solution.
        '''
        TINY = 1e-15
        m = self.Pi.shape[0]
        f0 = np.full(m, 1/m)
        for i in range(max_iter):
            q = self.Pi.T @ f0
            EM = f0 * (self.Pi@(p/(q+TINY)))
            f0[f0<=0] = 1 # in case np.log() sending warning
            E = lamb*(np.log(f0)-f0@np.log(f0))*f0
            f1 = EM - E
            if 2*norm(f1-f0)/(norm(f0)+norm(f1)+TINY) <= reltol:
                break
            else:
                f0 = f1
        status = 'success' if i<max_iter-1 else 'max_iter reached'
        self.message = {'status': status, 'num_iter': i+1}
        return f1