import numpy as np
from scipy.linalg import cho_solve
import numba

@numba.jit(nopython=True)
def p_distance_commutator(X,Y):
    a, b = len(X), len(Y)
    mat = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            d = X[i]@Y[j] - Y[j]@X[i]
            mat[i][j] = np.linalg.norm(d)
    return mat

def p_distance_eigval_batch(X, Y, batch_size=100):
    a, b = len(X), len(Y)
    mat = np.zeros((a,b))
    for i in range(0, a, batch_size):
        for j in range(0, b, batch_size):
            end_idx_i = min(i + batch_size, a)
            end_idx_j = min(j + batch_size, b)  
            eigvals_X = np.linalg.eigvals(X[i:end_idx_i])
            eigvals_Y = np.linalg.eigvals(Y[j:end_idx_j])
            eigvals_X = eigvals_X[:, np.newaxis, :]
            eigvals_Y = eigvals_Y[np.newaxis, :, :]
            d_eig = eigvals_X - eigvals_Y
            mat_batch = np.linalg.norm(d_eig, axis=2)
            mat[i:end_idx_i, j:end_idx_j] = mat_batch
    return mat

def KRR_commutator(X_train,Y_train,X_test,best_params,kernel='rbf',dist1='na',dist2='na'):
    lam = best_params['lambda']
    params = best_params
    if type(dist1)==str:
        dist1=p_distance_commutator(X_train,X_train)
    K=covariance(dist1,kernel,params)
    K+=(np.eye(K.shape[0])*lam)
    try:
        L=np.linalg.cholesky(K)
    except:
        return 'Gram Matrix is not positive definite'
    else:
        try:
            alpha=cho_solve((L,True),Y_train)
        except:
            return 'Cholesky decomposition failed, check distance matrices'
        else:
            if type(dist2)==str:
                dist2=p_distance_commutator(X_train,X_test)
            k=covariance(dist2,kernel,params)
            return np.dot(k.T,alpha)

def KRR_commutator_with_eig(X_train,Y_train,X_test,best_params,kernel='rbf',dist1='na',dist2='na'):
    lam = best_params['lambda']
    params = best_params
    if type(dist1)==str:
        dist1=p_distance_commutator(X_train,X_train) + p_distance_eigval_batch(X_train, X_train)
    K=covariance(dist1,kernel,params)
    K+=(np.eye(K.shape[0])*lam)
    try:
        L=np.linalg.cholesky(K)
    except:
        return 'Gram Matrix is not positive definite'
    else:
        try:
            alpha=cho_solve((L,True),Y_train)
        except:
            return 'Cholesky decomposition failed, check distance matrices'
        else:
            if type(dist2)==str:
                dist2=p_distance_commutator(X_train,X_test) + p_distance_eigval_batch(X_train, X_test)
            k=covariance(dist2,kernel,params)
            return np.dot(k.T,alpha)

def KRR_eig(X_train,Y_train,X_test,best_params,kernel='rbf',dist1='na',dist2='na'):
    lam = best_params['lambda']
    params = best_params
    if type(dist1)==str:
        dist1=p_distance_eigval_batch(X_train, X_train)
    K=covariance(dist1,kernel,params)
    K+=(np.eye(K.shape[0])*lam)
    try:
        L=np.linalg.cholesky(K)
    except:
        return 'Gram Matrix is not positive definite'
    else:
        try:
            alpha=cho_solve((L,True),Y_train)
        except:
            return 'Cholesky decomposition failed, check distance matrices'
        else:
            if type(dist2)==str:
                dist2=p_distance_eigval_batch(X_train, X_test)
            k=covariance(dist2,kernel,params)
            return np.dot(k.T,alpha)

@numba.jit(nopython=True)
def generate_CM(cood,charges,pad):
    size=len(charges)
    cm=np.zeros((pad,pad))
    for i in range(size):
        for j in range(size):
            if i==j:
                cm[i,j]=0.5*(charges[i]**(2.4))
            else:
                dist=np.linalg.norm(cood[i,:]-cood[j,:])
                
                cm[i,j]=(charges[i]*charges[j])/dist
    summation = np.array([sum(x**2) for x in cm])
    sorted_mat = cm[np.argsort(summation)[::-1,],:]    
    return sorted_mat

def covariance(dist,kernel,params):
    if kernel=='linear':
        K=(params['sigma0']**2)+((params['sigma1']**2)*dist)
        return K
    elif kernel=='polynomial':
        K=((params['sigma0']**2)+((params['sigma1']**2)*dist))**params['order']
        return K
    elif kernel in ['rbf','gaussian','Gaussian']:
        dist=dist/(params['length'])
        return np.exp(-(dist**2)/2)
    elif kernel=='laplacian':
        dist=dist/(params['length'])
        return np.exp(-dist)
    elif kernel=='matern1':
        dist=(3**0.5)*dist/(params['length'])
        return (1+dist)*np.exp(-dist)
    elif kernel=='matern2':
        dist1=(5**0.5)*dist/(params['length'])
        dist2=5*(dist**2)/(3*(params['length']**2))
        return (1+dist1+dist2)*np.exp(-dist)
    elif kernel=='rq':
        dist=(dist**2)/(2*params['alpha']*(params['length']**2))
        return (1+dist)**(-params['alpha'])