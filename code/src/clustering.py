import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def group_table_by_joints(table:pd.DataFrame) -> pd.DataFrame:
    """
    Groups columns in a DataFrame representing joint data into arrays per joint.
    
    Parameters:
    table (pd.DataFrame): Input DataFrame with joint data.
    
    Returns:
    pd.DataFrame: DataFrame with columns representing joints as arrays.
    """
    num_columns = len(table.columns)
    for i in range(0,num_columns, 3):
        table[table.columns[i][:-2]] = table.apply(lambda row: np.array([row.iloc[i], row.iloc[i+1], row.iloc[i+2]]), axis=1)
    return table.drop(table.columns[:num_columns], axis=1)

def joints_array_to_xyz_columns(table: pd.DataFrame) -> pd.DataFrame:
    """
    Converts arrays of joint data in columns to separate X, Y, Z columns for each joint.
    
    Parameters:
    table (pd.DataFrame): Input DataFrame with joint arrays in columns.
    
    Returns:
    pd.DataFrame: DataFrame with separate X, Y, Z columns for each joint.
    """
    return pd.concat([table[column].apply(lambda row: pd.Series(row,index=[column+'_X',column+'_Y',column+'_Z'])) for column in table.columns],axis=1)
    
def table_to_list_xyz_tables(table:pd.DataFrame,into="xyz"):
    """
    Converts DataFrame containing joint data into separate X, Y, Z tables or a list of point tables.
    
    Parameters:
    table (pd.DataFrame): Input DataFrame with joint data.
    into (str): Specifies the output format. Default is "xyz" to return separate X, Y, Z tables per joint. 
                If "points" is specified, returns a list of point tables with X, Y, Z columns.
    
    Returns:
    If into == "xyz":
        tuple: Three DataFrames representing X, Y, Z tables respectively.
    If into == "points":
        list: List of DataFrames, each representing a point with X, Y, Z columns.
    """
    if into == "xyz":
        return table.iloc[:,::3],table.iloc[:,1::3],table.iloc[:,2::3]
    elif into == "points":
        return [table.iloc[:,j:j+3] for j in range(0,table.shape[1],3)]

def xyz_tables_to_xyz_columns(tablesList):
    """
    Merges separate X, Y, Z tables into a single DataFrame with columns for X, Y, Z coordinates.
    
    Parameters:
    tablesList (list of pd.DataFrame): List containing separate X, Y, Z DataFrames for each joint.
    
    Returns:
    pd.DataFrame: DataFrame with columns for X, Y, Z coordinates of each joint.
    """
    xTable,yTable,zTable = tablesList
    mergedTable = pd.DataFrame()
    for joint in range(xTable.shape[1]):
        mergedTable = pd.concat([mergedTable,xTable.iloc[:,joint],yTable.iloc[:,joint],zTable.iloc[:,joint]],axis=1)
    return mergedTable


#################### reproducing MATLAB movmean and movmedian
def movmean(x, w):
    """
    Calculates the moving average of a series.
    
    Parameters:
    x (array-like): Input array or series.
    w (int): Window size for the moving average calculation.
    
    Returns:
    pd.Series: Series containing the moving average values.
    """
    return pd.Series(x).rolling(window=w,min_periods=1,center=True).mean()

from scipy.signal import medfilt

def movmedian(x,w):
    """
    Calculates the moving median of a sequence.
    
    Parameters:
    x (array-like): Input array or sequence.
    w (int): Window size for the moving median calculation.
    
    Returns:
    np.ndarray: Array containing the moving median values.
    """
    return np.concatenate([np.convolve(x[:w-1], np.ones(w-1), 'valid') / (w-1),medfilt(x,w)[1:-1],np.convolve(x[-w+1:], np.ones(w-1), 'valid') / (w-1)])
###################

################### features computation
def smoothing(x: pd.DataFrame) -> pd.DataFrame:
    """
    Performs smoothing on each column of a DataFrame using moving median followed by moving average.
    
    Parameters:
    x (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with smoothed values for each column.
    """
    return x.apply(lambda col: movmedian(col,5)).apply(lambda col: movmean(col,25))
    
def compute_derivatives(x:pd.DataFrame,dt:float,smooth=True) -> pd.DataFrame:
    """
    Computes derivatives of a DataFrame representing motion data.

    Parameters:
    x (pd.DataFrame): Input DataFrame containing motion data.
    dt (float): Time interval between samples.
    smooth (bool): Indicates whether to apply smoothing to the computed derivatives. Default is True.

    Returns:
    pd.DataFrame: DataFrame containing computed derivatives or smoothed derivatives if smooth is True.
    """
    v : pd.DataFrame
    v = x.diff() / dt
    v = v.fillna(v.iloc[1:10,:].mean())
    return smoothing(v) if smooth else v
###################

def calculate_weight_matrix(featuresTable: pd.DataFrame, adjacencyMatrix: np.ndarray,tol:float=None,use_cosine=True,include_non_physical=False,as_similarity=True) -> np.ndarray:
    """
    Calculate the weight matrix for a given set of features and adjacency matrix.

    Parameters:
        - featuresTable (pd.DataFrame): A DataFrame containing the features as rows.
        - adjacencyMatrix (np.ndarray): An adjacency matrix indicating connections between data points.
        - tol (float, optional): A tolerance value used for scalar features. Default is None.
        - use_cosine (bool, optional): Use cosine similarity for non-scalar features. Default is True. (better in every case)
        - as_similarity (bool, optional): Choose either similarity measure or distance measure (1 - similarity)

    Returns:
        - weight_matrix (np.ndarray): A matrix of weights reflecting the pairwise relationships between data points.

    Notes:
        - If features are non-scalar (e.g., vectors), cosine similarity is used by default.
        - If features are scalar (e.g., single values), you can specify a tolerance value (tol).
        - The weight matrix is symmetric and represents the strength of connections between data points.

    Example:
        # Calculate the weight matrix for non-scalar features with cosine similarity
        weight_matrix = calculate_weight_matrix(featuresTable, adjacencyMatrix)

        # Calculate the weight matrix for scalar features with a tolerance value
        weight_matrix = calculate_weight_matrix(featuresTable, adjacencyMatrix, tol=0.01, use_cosine=False)
    """
    if include_non_physical:
        adjacencyMatrix = np.ones(adjacencyMatrix.shape,bool)
    if isinstance(featuresTable.iloc[0],np.ndarray):
        values = np.stack(featuresTable.values)
        if not use_cosine:
            norm_matrix = np.linalg.norm(values, axis=1)
            norm_crossed_matrix = norm_matrix[:, None] * norm_matrix
            dot_crossed_matrix = np.dot(values, values.T)
            #                           where                       True                                                False
            weight_matrix = np.where(adjacencyMatrix, dot_crossed_matrix / norm_crossed_matrix + 1, dot_crossed_matrix / (norm_crossed_matrix + 1) * 1/5)
        else:
            weight_matrix = np.where(adjacencyMatrix, (cosine_similarity(values)+1)/2, 0)
            if not as_similarity: weight_matrix = 1 - weight_matrix
    else:
        if tol is None:
            raise Exception('must provide tol value with scalar feature')
        diff_matrix = featuresTable.values[:, None] - featuresTable.values
        # norm matrix not needed since features in our case are already scalar
        norm_matrix = np.abs(diff_matrix) #np.linalg.norm(diff_matrix, axis=-1)
        if not use_cosine:
            #                           where                   True                    False
            weight_matrix = np.where(adjacencyMatrix, 1 / (norm_matrix + tol)+5, 1 / (5 * norm_matrix + tol+2))
            if not as_similarity: weight_matrix = 1/weight_matrix
        else:
            weight_matrix = np.where(adjacencyMatrix, (cosine_similarity(norm_matrix)+1)/2, 0)
            if not as_similarity: weight_matrix = 1 - weight_matrix
    np.fill_diagonal(weight_matrix,0)
    return weight_matrix

# Spectral clustering using Shi and Malik algorithm
def shi_malik_spectral_clustering(weightMatrix:np.ndarray) -> np.array:
    """
    Performs spectral clustering using the Shi-Malik algorithm.

    Parameters:
    weightMatrix (np.ndarray): Weight matrix representing similarities between data points.

    Returns:
    np.array: Array containing cluster assignments based on spectral clustering.
    """
    # Compute the Laplacian matrix
    laplacian_matrix = np.diag(np.array(weightMatrix.sum(axis=0))) - weightMatrix
    
    # Compute eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    
    # Find the index of the second smallest eigenvalue
    idx = np.argsort(-eigenvalues)[1]
    
    # Extract new features from the corresponding eigenvector
    new_features = eigenvectors[:, idx]
    
    # Perform spectral clustering based on the sign of the new features
    return (new_features > 0).astype(int)


def myeigs(S: np.ndarray, unused_k=2):
    """
    Computes eigenvectors and eigenvalues of a given matrix.

    Parameters:
    S (np.ndarray): Input matrix.
    unused_k (int): Unused parameter in the function. Default value is 2.

    Returns:
    tuple: Tuple containing eigenvector and eigenvalue.
    """
    Sdiff = S - S.T  # Compute the difference between S and its transpose
    
    # Compute eigenvalues and eigenvectors of the matrix S
    eigenvalues, eigenvectors = np.linalg.eig(S)
    
    # Check if the matrix is symmetric
    if np.max(Sdiff) < 10**-20:
        raise ValueError("Symmetric matrix S, not implemented")
    
    # Find the index of the second largest eigenvalue
    idx = np.argsort(-eigenvalues)[1]
    
    # Return the corresponding eigenvector and eigenvalue
    return eigenvectors[:, idx].T, eigenvalues[idx]



def split_a_cluster(S, ww, nmin, optimizationOptions=None): # optimizationOptions is always ncut in this instance
    """
    Splits a cluster using some magic tricks of permutations.

    Parameters:
    S (np.ndarray): Input matrix.
    ww (array-like): Array containing cluster information.
    nmin (int): Minimum number of elements required in a cluster.
    optimizationOptions (str or None): Specifies the optimization option. Default is None.

    Returns:
    tuple: Tuple containing cluster elements, eigenvalues, and cluster information after splitting.
    """
    n = len(ww)
    tt = np.sum(S,axis=0)
    iPerm = np.argsort(ww)
    #ww = ww[iperm] ww not used anymore in this instance

    nCuts = np.zeros(n-1)
    for iCut in range(n-1):
        i1 = np.arange(iCut+1)
        i2 = np.arange(iCut+1,n)
        i1 = iPerm[i1]
        i2 = iPerm[i2]
        mult_factor = 1/np.sum(tt[i1]) + 1/np.sum(tt[i2])
        nCuts[iCut] = np.sum(S[i1,:][:,i2]) * mult_factor
    iMin = np.argmin(nCuts)
    i1 = np.arange(iMin+1)
    i2 = np.arange(iMin+1,n)
    elems1 = iPerm[i1]
    elems2 = iPerm[i2]

    if len(i1) > nmin:
        S1 = S[elems1,:][:,elems1]
        T1 = np.diag(np.sum(S1,axis=0))
        P1 = np.linalg.lstsq(T1,S1,rcond=None)[0]
        ww1, lambda1 =  myeigs(P1,2)
    else:
        lambda1 = -2
        ww1 = np.ones(len(i1))

    if len(i2) > nmin:
        S2 = S[elems2,:][:,elems2]
        T2 = np.diag(np.sum(S2,axis=0))
        P2 = np.linalg.lstsq(T2,S2,rcond=None)[0]
        ww2,lambda2 = myeigs(P2,2)
    else:
        lambda2 = -2
        ww2 = np.ones((len(i2)))
    return elems1,elems2,lambda1,lambda2,ww1,ww2


def shi_malik_spectral_clustering_matlab_version(S: np.ndarray, k:int=4):
    """
    Performs spectral clustering using the Shi-Malik algorithm.

    Parameters:
    S (np.ndarray): Input matrix.
    k (int): Number of clusters to generate. Default is 4.

    Returns:
    np.ndarray: Array containing cluster assignments based on spectral clustering.
    """
    n = len(S)
    T = np.sum(S,axis=1)
    P = S / np.tile(T[:,np.newaxis],len(T))
    nmin = 5
    KMAX = min(10,2*k)

    shi_nclust = 0
    shi_ww = np.zeros((KMAX,n))
    shi_elems = np.zeros((KMAX,n),dtype=int)
    shi_n = np.zeros(KMAX,dtype=int)
    shi_lambda = np.zeros(KMAX)
    assignment_shi_r = np.zeros(n,dtype=int)

    shi_elems[0,:] = np.arange(n)
    shi_n[0] = n
    
    isplit = 0
    shi_ww[0,:] = myeigs(P,2)[0]
    while shi_nclust < k-1: 
        elems = shi_elems[isplit,:shi_n[isplit]].copy()
        elems1,elems2,lambda1,lambda2,ww1,ww2 = split_a_cluster(S[elems,:][:,elems],shi_ww[isplit,:shi_n[isplit]],nmin,None)
        nn1 = len(elems1)
        nn2 = len(elems2)
        shi_nclust+=1
        shi_elems[isplit,:nn1] = elems[elems1]
        shi_elems[shi_nclust,:nn2] = elems[elems2] 
        shi_lambda[isplit] = lambda1
        shi_lambda[shi_nclust] = lambda2
        shi_ww[isplit,:nn1] = ww1
        shi_ww[shi_nclust,:nn2] = ww2
        shi_n[isplit] = nn1
        shi_n[shi_nclust] = nn2
        ldummy, isplit = np.max(shi_lambda[:shi_nclust+1]), np.argmax(shi_lambda[:shi_nclust+1])
        if ldummy == -2:
            print('cant find k clusters with these conditions')
    for iclust in range(k):
        assignment_shi_r[shi_elems[iclust,:shi_n[iclust]]] = iclust
    return assignment_shi_r


from pandas import Series
from numpy import array, sum
from itertools import permutations
from typing import Literal
from networkx.algorithms.bipartite import minimum_weight_full_matching as min_wei_ful_match
import networkx as nx
from matplotlib import pyplot as plt

def compute_cost(fromLabels:list or dict,toLabels:list or dict):
    """
    Computes the cost based on label differences between two label lists.

    Parameters:
    fromLabels (list): List of original labels.
    toLabels (list): List of new labels.
    labels_kind (Literal['full', 'cluster']): Specifies the kind of labels comparison. Default is 'full'.

    Returns:
    int: Cost computed based on label differences between the two label lists.
    
    Raises:
    Exception: If the provided method is not implemented.
    """
    if isinstance(fromLabels,list) or isinstance(fromLabels,np.ndarray) :
        assert len(fromLabels) == len(toLabels)
        return sum(array(fromLabels) != array(toLabels))
    elif isinstance(fromLabels,dict):
        return len(set(toLabels).difference(set(fromLabels)))
    raise Exception("method not implemented")



def clusterize_labels(labels:list) -> dict:
    """
    Groups labels into clusters based on integer values ranging from 0 to n.

    Parameters:
    labels (list): List of integers ranging from 0 to n.

    Returns:
    dict: Dictionary with keys representing clusters and values as lists containing labels in each cluster.
    """
    df = Series(labels)
    clusters = df.groupby(df).groups
    keys = set(clusters.keys())
    for i in range(max(keys)):
        if i not in keys:
            clusters[i] = []
    return clusters



def labelize_clusters(clusters:dict):
    """
    Assigns cluster labels to nodes based on the given clusters.

    Parameters:
    clusters (dict): Dictionary containing clusters with node indices.

    Returns:
    np.ndarray: Array containing cluster labels assigned to nodes.
    """
    clusterLabels = np.zeros(sum(len(nodes) for nodes in clusters.values()), dtype=int)
    for clusterLabel, indices in clusters.items():
        clusterLabels[indices] = clusterLabel
    return clusterLabels


def compute_minimum_weight_cluster(fromLabels:list,toLabels:list,method:Literal['BF','MWPM']="BF",visualize=False,return_relabeling=False,return_cost=False) -> tuple :
    """
    Computes the minimum weight cluster based on different methods.

    Parameters:
    fromLabels (list): List of original labels.
    toLabels (list): List of new labels.
    method (Literal['BF', 'MWPM']): Method for computing minimum weight cluster. Default is 'BF'.
    visualize (bool): Indicates whether to visualize the process. Default is False.
    return_relabeling (bool): Indicates whether to return the relabeling information. Default is False.

    Returns:
    tuple: Tuple containing elements representing cluster labels or relabeling information based on the method used.
    
    Raises:
    ValueError: If an unsupported method is provided.
    """

    # Reassign labels 
    out_elements = []

    # start by bruteforce if different number of clusters
    if len(set(fromLabels)) != len(set(toLabels)):
        toClusters = clusterize_labels(toLabels)
        clusterLabelPermutations = list(permutations(list(toClusters.keys())))
        toClustersValues = list(toClusters.values())
        optimalLabelingIndex = -1
        minCost = 10**5
        # finds optimal permutation
        for indexPermutation,clusterLabelPermutation in enumerate(clusterLabelPermutations):
            permCost = compute_cost(fromLabels,labelize_clusters(dict(zip(clusterLabelPermutation,toClustersValues))))
            if permCost < minCost:
                optimalLabelingIndex = indexPermutation
                minCost = permCost
        return labelize_clusters(dict(zip(clusterLabelPermutations[optimalLabelingIndex],toClustersValues)))


    # apply Bruteforce
    if method.upper() == "BF":
        toClusters = clusterize_labels(toLabels)
        clusterLabelPermutations = list(permutations(list(toClusters.keys())))
        toClustersValues = list(toClusters.values())
        optimalLabelingIndex = -1
        minCost = 10**5
        if visualize:
            leftNodes = [0]
            rightNodes = list(range(1,len(clusterLabelPermutations)+1))
            plt.figure(figsize=(8,8))
            G = nx.DiGraph()
            G.add_nodes_from(leftNodes,bipartite=0,color='#1f78b4')
            G.add_nodes_from(rightNodes,bipartite=1,color='#33a02c')
            pos = nx.bipartite_layout(G,leftNodes)
            x_center = pos[leftNodes[0]][0]
            y_center = pos[leftNodes[0]][1]
            for node in rightNodes:
                angle = (2 * node * 3.14159) / len(rightNodes)  # Distribute the right nodes evenly around the center node
                distance = 2.0  # Adjust this value to control the distance of the right nodes from the center node
                x = x_center + distance * np.cos(angle)
                y = y_center + distance * np.sin(angle)
                pos[node] = (x, y)
        
        # finds optimal permutation
        for indexPermutation,clusterLabelPermutation in enumerate(clusterLabelPermutations):
            permCost = compute_cost(fromLabels,labelize_clusters(dict(zip(clusterLabelPermutation,toClustersValues))))
            if visualize:
                G.add_edge(leftNodes[0],rightNodes[indexPermutation],weight=permCost,color='b',width=1)
            if permCost < minCost:
                optimalLabelingIndex = indexPermutation
                minCost = permCost

        if visualize:
            edgeColors = ['b' if i == optimalLabelingIndex else '#C4C2C6' for i in range(len(rightNodes))]
            edgeWidths = [3 if i == optimalLabelingIndex else 1 for i in range(len(rightNodes))]
            nx.draw(G, pos=pos,with_labels=True,node_size=900,node_color=list(nx.get_node_attributes(G, 'color').values()),edge_color=edgeColors,width=edgeWidths)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G,'weight'), label_pos=0.5)
        out_elements.append(labelize_clusters(dict(zip(clusterLabelPermutations[optimalLabelingIndex],toClustersValues))))
        if return_relabeling:
            out_elements.append(clusterLabelPermutations[optimalLabelingIndex])
        if return_cost:
            out_elements.append(compute_cost(fromLabels,labelize_clusters(dict(zip(clusterLabelPermutations[optimalLabelingIndex],toClustersValues)))))
    elif method.upper() == 'MWPM':
        fromClustersMap = clusterize_labels(fromLabels)
        clusters = list(fromClustersMap.keys())
        toClustersUnlabelized = list(clusterize_labels(toLabels).values())
        rightNodes = (np.array(clusters) + max(clusters)+1).tolist()
        #print(rightNodes)

        edgeWeights = {fromLabel : 
                            {toLabelAssigned+max(clusters)+1 : 
                                {'weight': compute_cost(fromClustersMap[fromLabel],toClustersUnlabelized[toLabelAssigned])}
                             for toLabelAssigned in clusters } 
                         for fromLabel in clusters }
        
        G = nx.from_dict_of_dicts(edgeWeights,create_using=nx.DiGraph)
        
        minWeightsEdges = {fromNode:toNode for fromNode, toNode in min_wei_ful_match(G,clusters,'weight').items() if fromNode not in rightNodes}
        reversedWeightsEdgesDict = {value:key for key,value in minWeightsEdges.items()}
        if visualize:
            color_pool = ['#e31a1c', '#1f78b4', '#33a02c', '#ff7f00', '#6a3d9a', '#b15928']
            plt.figure(figsize=(11,8))
            pos = nx.bipartite_layout(G, clusters)
            nx.draw(G, 
                    pos,
                    with_labels=True,
                    labels={elemFrom:elemFrom%len(clusters) for elemFrom in clusters+rightNodes}, 
                    node_size=1000, 
                    node_color= [color_pool[nodeId] for nodeId in clusters] +
                                [color_pool[reversedWeightsEdgesDict[toNodeId]] for toNodeId in sorted(reversedWeightsEdgesDict.keys())], 
                    font_size=10,
                    edge_color=[color_pool[edge[0]] if edge in list(minWeightsEdges.items()) else '#C4C2C6' for edge in G.edges()],
                    width=[3 if edge in list(minWeightsEdges.items()) else 1 for edge in G.edges()])
            nx.draw_networkx_edge_labels(G, 
                                         pos, 
                                         edge_labels=nx.get_edge_attributes(G, 'weight'),
                                         label_pos=0.85,
                                         font_size=10)#,
                                         #font_weight='bold')
            for node in G.nodes:
                text = list(fromClustersMap[node]) if node <= max(clusters) else list(toClustersUnlabelized[node-max(clusters)-1])
                text = list(map(str,text))
                text = '[ '+',\n'.join(', '.join(text[i:i+3]) for i in range(0, len(text), 3))+' ]'
                
                plt.text(pos[node][0], 
                         pos[node][1]+0.11, 
                         text,#str(list(fromClustersMap[node]))[1:-1] if node <= max(clusters) else str(list(toClustersUnlabelized[node-max(clusters)-1]))[1:-1], 
                         ha='center', 
                         va='center', 
                         color='black', 
                         fontsize=10,
                         fontweight='bold')
            #plt.suptitle('Minimum weight clusters',fontsize=14,fontweight='bold')
            plt.show()
        out_elements.append(labelize_clusters({fromNode: toClustersUnlabelized[minWeightsEdges[fromNode]-max(clusters)-1] for fromNode in minWeightsEdges.keys()}))
        if return_relabeling:
            out_elements.append({fromNode: minWeightsEdges[fromNode]-max(clusters)-1 for fromNode in minWeightsEdges.keys()})
        if return_cost:
            out_elements.append(compute_cost(fromLabels,labelize_clusters({fromNode: toClustersUnlabelized[minWeightsEdges[fromNode]-max(clusters)-1] for fromNode in minWeightsEdges.keys()})))
    return out_elements if len(out_elements) > 1 else out_elements[0]



def compute_auxiliary_graph(featuresTable:pd.DataFrame,clusters:np.ndarray,adjacencyMatrix:np.ndarray) -> np.ndarray:
    """
    Computes an auxiliary graph based on the features table, clusters, and adjacency matrix.

    Parameters:
    featuresTable (pd.DataFrame): DataFrame containing features.
    clusters (np.ndarray): Array representing clusters.
    adjacencyMatrix (np.ndarray): Adjacency matrix.

    Returns:
    np.ndarray: Computed auxiliary graph.
    """
    auxiliary_graph = np.zeros(adjacencyMatrix.shape)
    if isinstance(featuresTable.iloc[0],np.ndarray):
        vectorsArray = np.stack(featuresTable.values)
        normMatrix = np.linalg.norm(vectorsArray[:,None] - vectorsArray, axis=2)
    else:
        normMatrix = np.abs(featuresTable.values[:,None] - featuresTable.values)
    mask = np.logical_and(adjacencyMatrix,clusters[:,None] != clusters)
    auxiliary_graph[mask] = normMatrix[mask]
    return auxiliary_graph


def calculate_shapley_values(auxiliaryGraph:np.ndarray):
    """
    Calculates Shapley values based on the provided auxiliary graph.

    Parameters:
    auxiliaryGraph (np.ndarray): Auxiliary graph for Shapley value calculation.

    Returns:
    tuple: Tuple containing Shapley values, normalized Shapley values (scaled to maximum),
           and normalized Shapley values (scaled to the utility norm factor).
    """
    shapleys = 0.5 * np.sum(auxiliaryGraph,axis=1)
    maxShapley = np.max(shapleys)
    utilityNormFactor = np.sum(shapleys)
    return shapleys, shapleys / maxShapley, shapleys / utilityNormFactor # should also return mean across every time instant