import numpy as np




def compute_matrix_power(matrix, n):
    '''The resulting matrix gives at position (i,j) the number of shortest path of length n from i to j'''
    return np.linalg.matrix_power(matrix, n+1)


def compute_path_matrix(matrix, largest_path_length):
    return sum([compute_matrix_power(matrix, i) for i in range(largest_path_length)])


def compute_diameter(A_matrix, max_test=10):
    '''Compute the diameter of a adjacency matrix'''
    #Test that A_matrix is connected:
    assert np.sum(A_matrix, axis=1).min() != 0, "The graph is not connected"
    if A_matrix.min() != 0:
        return 1
    else:
        for i in range(2, max_test+1):
            if compute_path_matrix(A_matrix, i).min() != 0:
                return i
            else: continue

             
def compute_sigma_knn(dist, k):
    '''
        Compute the mean of the k nearest neighbour distance for each points in the dataset
    '''
    mean_dists = []
    for rm_point in range(2000):
        # Indices of the knn without zero
        n_nearest_idx = np.argsort(dist[rm_point,:])[:k]
        non_zero_idx = np.argwhere(dist[rm_point,:] != 0)
        mean_dists.append(dist[rm_point,\
                            np.intersect1d(n_nearest_idx, non_zero_idx)].mean())
        
    return np.array(mean_dists).mean()



def epsilon_similarity_graph(X: np.ndarray, sigma=None, epsilon=0, k=10):
    """ X (n x d): coordinates of the n data points in R^d.
        sigma (float): width of the kernel
        epsilon (float): threshold
        Return:
        adjacency (n x n ndarray): adjacency matrix of the graph.
    """
    dist = squareform(pdist(X.copy()))
    sig_tmp = sigma
    if sig_tmp is None:
        sig_tmp = compute_sigma_knn(dist, k)
        
    kernel = np.exp(-dist ** 2 / (2 * sig_tmp ** 2))
    if epsilon is None:
        epsilon = kernel.mean()
    adjacency = kernel.copy()
    adjacency[adjacency < epsilon] = 0
    np.fill_diagonal(adjacency, 0)
    print("Sigma: %f" % sig_tmp)
    return adjacency


def weights_histogram(adj):
    vals, bins = np.histogram(adj)
#     plt.bar(bins[2:], vals[1:], width=np.diff(bins[2:]), ec="k", align="edge")
    plt.plot(bins[2:], vals[1:])
    
    
def compute_laplacian(adjacency: np.ndarray, normalize: bool):
    """ Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    L = adjacency.copy()
    np.fill_diagonal(L, 0)
    diag = L.sum(axis=0)
    
    # Normalization
    if normalize:
        mask = (diag == 0)
        diag = np.where(mask, 1, np.sqrt(diag))
        L /= diag
        L /= diag[:, np.newaxis]
        L *= -1
        L.flat[::len(1-mask)+1] = 1 - mask
    else:
        L *= -1
        L.flat[::len(diag)+1] = diag
    return L


def spectral_decomposition(laplacian: np.ndarray):
    """ Return:
        lamb (np.array): eigenvalues of the Laplacian
        U (np.ndarray): corresponding eigenvectors.
    """
    Lambda, U = np.linalg.eigh(laplacian)
    idx = Lambda.argsort()
    Lambda = Lambda[idx]
    U = U[:, idx]
    return Lambda, U