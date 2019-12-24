import numpy as np
import sklearn.preprocessing
def camera_info(param):
    theta=np.deg2rad(param[0])
    phi=np.deg2rad(param[1])

    camY=param[3]*np.sin(phi)
    temp=param[3]*np.cos(phi)
    camX=temp*np.cos(theta)
    camZ=temp*np.sin(theta)
    cam_pos=np.array([camX,camY,camZ])

    axisZ=cam_pos.copy()
    axisY=np.array([0,1,0])
    axisX=np.cross(axisY,axisZ)
    axisY=np.cross(axisX,axisZ)

    cam_mat=np.array([axisX,axisY,axisZ])
    cam_mat=sklearn.preprocessing.normalize(cam_mat,axis=1)
    return cam_mat,cam_pos

def sample_triangle(v,n=None):
    if hasattr(n,'dtype'):
        n=np.asscalar(n)

    if n is None:
        size=v.shape[:-2]+(2,)
    elif isinstance(n,int):
        size=(n,2)
    elif isinstance(n,tuple):
        size=n+(2,)
    elif isinstance(n,list):
        size=tuple(n)+(2,)
    else:
        raise TypeError('n must be int, tuple or  list,got %s'%str(n))
    assert (v.shape[-2]==2)
    a=np.random.uniform(size=size)
    mask=np.sum(a,axis=-1)>1
    a[mask]*=-1
    a[mask]+=1
    a=np.expand_dims(a,axis=-1)
    return np.sum(a*v,axis=-2)

def sample_faces(vertices,faces,n,total):
    if len(faces)==0:
        raise  ValueError('cannot sample points from zero faces')
    tris=vertices[faces]
    n_faces=len(faces)
    d0=tris[...,0:1,:]
    ds=tris[...,1:,:]-d0
    assert(ds.shape[1:]==(2,3))
    areas=0.5*np.sqrt(np.sum(np.cross(ds[:,0],ds[:,1])**2,axis=-1))
    cum_area = np.cumsum(areas)
    cum_area *= (n_total / cum_area[-1])
    cum_area = np.round(cum_area).astype(np.int32)

    positions = []
    last = 0
    for i in range(n_faces):
        n = cum_area[i] - last
        last = cum_area[i]
        if n > 0:
            positions.append(d0[i] + sample_triangle(ds[i], n))
    return np.concatenate(positions, axis=0)

