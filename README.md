# Optimized detector tomography for photon-number-resolving detectors with hundreds of pixels
#### Description
1. We proposed a modified form for detector romography, see ```Tomography-of-quantum-detector.ipynb```
$$\hspace{12em}\mathrm{min}\quad \frac{1}{2}\|P-F\varPi\|^2_{\mathrm{Fro}} + \frac{\gamma}{2}\sum_{k=0}^{M-1}\sum_{n=0}^{N}(\varPi_{k,n}-\varPi_{k+1,n})^2$$
$$\mathrm{s.t.}\quad \varPi\boldsymbol{1}_ {N+1}=\boldsymbol{1}_ {M+1},$$
$$\hspace{5em} \varPi_{k,n}=0, \ \mathrm{if}\ \widetilde{\varPi}_ {k,n}\le0, $$
$$\hspace{5em} \varPi_{k,n}\ge 0, \ \mathrm{if}\ \widetilde{\varPi}_ {k,n}>0, $$
which reduces the degrees of freedom. The solving time is therefore reduced compared to that of standard detector tomography.
2. We numerically characterize the computational resources needed for detector tomography. The memory consumption is shown to be the main obstacle for detector tomography
