# benchmark time and memory consumption scaling with the number of detector pixels
import numpy as np
import pickle, os
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from memory_profiler import memory_usage
from functools import partial
import utils, RQS

# Generate coherent states as probe states for detector tomography, step length of nbar is 1.
# Detectors with different number of pixels are simulated to find out the time complexity of the detector tomography and state reconstruction.

def get_probe(N: int, threshold: float = .9) -> None:
    '''
    Get probe states statistics.
    Args:
        N: int, the number of pixels of the detector.
        threshold: float, increase the mean photon number of the probe states until the probability of N clicks is greater than the thershold, default 0.9
    '''
    detector = utils.Monte_Carlo(N)
    path = f'./Data/detector_{N}'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'detector.pickle'), 'wb') as f:
        pickle.dump(detector, f)

    P = []
    i = 1
    while True:
        P.append(detector.statistics('coherent', i))
        if P[-1][-1] >= threshold:
            break
        i += 1
    np.save(os.path.join(path, 'P_coherent.npy'), np.array(P))

def CVX(N: int, improved: bool, gamma: float = 1e-4, **kwargs) -> None:
    '''
    Detector tomography by convex optimization.
    Args:
        N: int, the number of pixels of the detector.
        improved: bool, whether to use the improved detector tomography.
        gamma: float, the regularizaiton parameter, default 1e-4.
        kwargs: kwargs for CVXPY solve method.
    '''
    path = f'./Data/detector_{N}'
    P = np.load(os.path.join(path, 'P_coherent.npy'))
    dt = RQS.dt_CVX(P, gamma, improved)
    
    base_line = memory_usage(timeout=1)[0]
    mem_usage = memory_usage((dt.prob.solve, (), kwargs), interval=10, max_usage=True, include_children=True) - base_line
    solve_time = dt.prob.solver_stats.solve_time
    if dt.prob.status != 'optimal':
        tqdm.write(f'{dt.prob.status=} for {N=}')
    
    if improved:
        save_Pi = 'Pi_improved.npy' # the file name to save Pi as.
        save_TimeAndMemory = 'TimeAndMemory_imporved.npy' # the file name to save time and memory consumption as.
    else:
        save_Pi = 'Pi.npy'
        save_TimeAndMemory = 'TimeAndMemory.npy'
    np.save(os.path.join(path, save_Pi), dt.Pi.value)
    np.save(os.path.join(path, save_TimeAndMemory), np.array([solve_time, mem_usage]))
    tqdm.write(f'{solve_time=:.2f}s, {mem_usage=:.2f}MiB')

def PM(N: int, y: float = .1) -> None:
    '''
    Detector tomography by parametric model.
    Args:
        N: int, the number of pixels of the detector.
        y: float, the regularizaiton parameter, default 0.1
    '''
    path = f'./Data/detector_{N}'
    P = np.load(os.path.join(path, 'P_coherent.npy'))
    mdt = RQS.dt_PM(P)
    
    base_line = memory_usage(timeout=1)[0]
    mem_usage = memory_usage((mdt.solve, (y,)), max_usage=True) - base_line
    # if res.message != 'optimal':
    #     tqdm.write(f'{dt.prob.status=} for {N=}')
    
    np.save(os.path.join(path, 'Pi_parametric.npy'), mdt.Pi(mdt.res.x))
    np.save(os.path.join(path, 'TimeAndMemory_parametric.npy'), np.array([mdt.solve_time, mem_usage]))
    tqdm.write(f'{mdt.solve_time=:.2f}s, {mem_usage=:.2f}MiB')
    
if __name__=='__main__':
    # Simulate measurement data
    process_map(get_probe, range(210, 251, 10))
    
    ## benchmark detector tomography with solver 'ECOS', which is quite slow.
    # process_map(partial(CVX, .1, 'Pi.npy', 'TimeAndMemory.npy', solver='ECOS'), range(10, 101, 10))
    
    # benchmark detector tomography with solver 'MOSEK', which supports multi-threading.
    for N in tqdm(range(10, 171, 10)):
        CVX(N, improved=True, solver='MOSEK', mosek_params={"MSK_IPAR_INTPNT_SOLVE_FORM":2}) #tell MOSEK that it should solve the dual
        CVX(N, improved=False, solver='MOSEK', mosek_params={"MSK_IPAR_INTPNT_SOLVE_FORM":2})
                               
    ## benchmark parametric detector tomography
    # N = np.arange(10, 51, 10)
    # process_map(PM, N)