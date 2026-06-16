import numpy as np, scipy.io as sio
N=(48,52,50)  # non-16x on purpose -> tests crop_img_16x (48,48,48)
g=np.meshgrid(*[np.linspace(-1,1,n) for n in N], indexing='ij')
r=np.sqrt(sum(x**2 for x in g))
mask=(r<0.85).astype(np.float32)
sio.savemat('in_r2p.mat', dict(
    mask=mask,
    local_field_hz=(5*np.exp(-(r/0.4)**2)*mask).astype(np.float32),
    r2star_hz=((15+25*np.exp(-(r/0.3)**2))*mask).astype(np.float32),
    r2_hz=((10+5*np.exp(-(r/0.5)**2))*mask).astype(np.float32),
))
sio.savemat('in_r2s.mat', dict(
    mask=mask,
    local_field_hz=(5*np.exp(-(r/0.4)**2)*mask).astype(np.float32),
    r2star_hz=((15+25*np.exp(-(r/0.3)**2))*mask).astype(np.float32),
))
print('inputs written, shape', N)
