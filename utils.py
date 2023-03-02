# cython: language_level=3

import numpy as np
import numpy.typing as npt
from typing import Optional
from itertools import product
from scipy import stats
from scipy.special import gammaln
rng = np.random.default_rng()

class _squeezed_vacuum_gen(stats.rv_discrete):
    '''Random generator for squeezed vacuum state distribution.'''
    def _logpmf(self, k, mu):
        s = np.arcsinh(np.sqrt(mu))
        return -np.log(np.cosh(s)) + gammaln(k+1) - 2*gammaln(k//2+1) + k*np.log(1/2*np.tanh(s))
    
    def _pmf(self, k, mu):
        k = np.array(k)
        rho = np.zeros(len(k))
        for i in range(len(k)):  
            if k[i]%2==0:
                rho[i] = np.exp(self._logpmf(k[i], mu))
        return rho
    
    # The sampling is quite slow.

squeezed_vacuum = _squeezed_vacuum_gen(name='squeezed_vacuum')


class Monte_Carlo():
    def __init__(self, N: int, num_pulse: int = int(1e5)):
        '''
        Args:
            N: int, number of detectors.
            num_pulse: int, number of pulses, default int(1e5).
        Additional attributes:
            eta_c: float, coupling efficiency of the detector, default 0.99.
            eta_a: 2darray, absorption efficiency of each pixel.
            eta_i: 2darray, intrinsic efficiency of each pixel.
        '''
        self.N = N
        self.num_pulse = num_pulse
        self.eta_c = .99
        self.eta_a = np.array([1/2*1/(N//2-n) for n in range(N//2)]*2).reshape(2, -1)
        self.eta_a = rng.normal(self.eta_a, .02*self.eta_a)
        self.eta_i = rng.uniform(.9, .95, (2, N//2))

    def statistics(self, state: str, nbar: float) -> npt.NDArray[float]:
        '''
        Simulate the measured statistics. 
        Args:
            state: str, one of 'coherent', 'thermal', 'squeezed vacuum', 'fock'.
            nbar (float): average number of photons.
        Return:
            click_hist: 1darray of shape (N+1,), measured statistics.
        '''
        pulses = self.get_pulses(state, nbar)
        clicks = [self.single_shot(k) for k in pulses]
        click_hist, _ = np.histogram(clicks, bins=np.arange(self.N+2), density=True)
        return click_hist
    
    def get_pulses(self, state: str, nbar: float) -> npt.NDArray[float]:
        '''
        Sample from the state to get the number of photons in each pulse.
        Args:
            state: str, one of 'coherent', 'thermal', 'squeezed vacuum', 'fock'.
            nbar (float): average number of photons.
        Return:
            pulses: 1darray, the number of photons in each pulse
        '''
        if state=='coherent':
            nbar_ = rng.normal(loc=nbar, scale=.0188*nbar, size=self.num_pulse) # consider the technical noise of the laser.
            pulses = rng.poisson(nbar_) # the number of photons in each pulse.
        elif state=='thermal':
            pulses = rng.geometric(1/(1+nbar), size=self.num_pulse) - 1
        elif state=='squeezed vacuum':
            pulses = squeezed_vacuum.rvs(nbar, size=self.num_pulse)
        elif state=='fock':
            pulses = np.full(self.num_pulse, nbar, dtype=int)
        else:
            raise ValueError("state should be one of 'coherent', 'thermal', 'squeezed vacuum', 'fock'.")
        return pulses
        
    def single_shot(self, k: int) -> int:
        '''
        Compute the number of clicks when k photons incidents. 
        Args:
            k: int, number of photons incident on the detector.
        Return:
            num_click: int, number of clicks when k photons incident on the detector.
        '''
        if k==0:
            return 0
        else:
            k = rng.binomial(k, self.eta_c) # coupling loss
            if k==0:
                return 0
            else:
                pixel_status = np.zeros((2, self.N//2))
                for j, i in product(range(self.N//2), range(2)):
                    absorbed = rng.binomial(k, self.eta_a[i, j]) # absorption by each pixel
                    if absorbed != 0:
                        pixel_status[i, j] = rng.binomial(absorbed, self.eta_i[i, j]) # intrinsic efficiency of each pixel
                        k -= absorbed
                        if k==0:
                            break
                num_click = np.count_nonzero(pixel_status)
                return num_click

def rho(mu: float, M: int, state_type: str, normalize=True) -> npt.NDArray[float]:
    '''
    The probability mass function of the distribution.
    Parameters:
        mu: float, average photon number.
        M: int, the number state |M> at which rho_ is truncated.
        state_type: str, the type of the state, must be one of 'coherent', 'thermal', 'squeezed vacuum', and 'fock'.
    Returns:
        rho_: 1darray, the state truncated at M.
    Refer to Loudon, R. The quantum theory of light. (Oxford University Press, 2000).
    '''
    if state_type=='coherent':
        rho_ = stats.poisson.pmf(np.arange(M+1), mu)
    elif state_type=='thermal':
        rho_ = stats.geom.pmf(np.arange(M+1)+1, 1/(1+mu))
    elif state_type=='squeezed vacuum':
        rho_ = squeezed_vacuum.pmf(np.arange(M+1), mu)
    elif state_type=='fock':
        rho_ = np.zeros(M+1)
        rho_[int(mu)] = 1
    else:
        raise ValueError("state_type should be one of 'coherent', 'thermal', 'squeezed vacuum', 'fock'")
    
    if normalize and state_type!='fock':
        rho_ /= rho_.sum()
    return rho_

def get_M(nbar_max: float, threshold: float = 1e-5) -> int:
    '''
    Determine the truncation number M.
    Args:
        nbar_max: float, the maximum mean number of photons of the probe states.
        threshold: float, default 1e-5. Truncate the Fock space at M, where the probability of the poisson distriburion is smaller than the threshold.
    Return:
        M: int, the number state at which the Fock space is truncated.
    '''
    M = int(nbar_max)
    p = stats.poisson.pmf(M, nbar_max)
    while p > threshold:
        M += 1
        p = stats.poisson.pmf(M, nbar_max)
    return M

def get_F(nbar: npt.NDArray[float], M: int, state_type: str):
    '''
    Args:
        nbar: 1darray, the mean photon number of the states.
        M: int, the number state at which the states truncate.
        state_type: str, the type of the state, must be one of 'coherent' and 'squeezed vacuum'.
    Return:
        F: 2darray, the input states.
    '''
    F = np.empty((len(nbar), M+1))
    for i, mu in enumerate(nbar):
        F[i] = rho(mu, M, state_type)
    return F

def fidelity(p: npt.NDArray[float], q: npt.NDArray[float]) -> float:
    '''
    Compute the classic fidelity of two states.
    Args:
        p: ndarray, a distribution.
        q: ndarray, another distribution.
    Return:
        fid: float, the fidelity.
    '''
    return (np.sqrt(p) @ np.sqrt(q))**2

def TVD(p: npt.NDArray[float], q: npt.NDArray[float]) -> float:
    '''
    Compute the total variation distance.
    Args:
        p: ndarray, a distribution.
        q: ndarray, another distribution.
    Return:
        tvd: float, the total variaton distance.
    '''
    return np.linalg.norm(p-q, ord=1) / 2

def gn(ord: int, p: npt.NDArray[float]) -> float:
    '''
    Compute the correlation function $g^{(ord)}(0)$ with order ord.
    $$g^{(ord)}(0) = \frac{\langel n(n-1)\cdots[n-(ord-1)] \rangle}{\langle n \rangle^{ord}}$$
    Args:
        ord: int, the order of the correlation function.
        p: (1darry) the distribution of which the correlation function is to be computed.
    Returns:
        g: float, the correlation function with order ord.
    '''
    n = np.arange(len(p))
    nbar = n@p
    g = 1/nbar**ord
    for i in range(ord):
        g *= n-i
    g = g@p
    return g