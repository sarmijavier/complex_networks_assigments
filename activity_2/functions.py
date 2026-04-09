
from infomap import Infomap
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.metrics import mutual_info_score
import numpy as np

def entropy(labels):
    probs = np.bincount(labels) / len(labels)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def normalized_vi(labels_true, labels_pred):
    n = len(labels_true)
    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)
    mi = mutual_info_score(labels_true, labels_pred)
    
    vi = h_true + h_pred - 2 * mi
    return vi / np.log(n)

def infomap_to_networkx_communities(G):
    im = Infomap(silent=True) 
    mapping = im.add_networkx_graph(G)
    im.run()

    infomap_partition = im.get_modules()
    community_dict = defaultdict(set)
    for node, community in infomap_partition.items():
        community_dict[community].add(mapping[node])
    
    return list(community_dict.values())

def community_jaccard(labels_true, labels_pred):
    
    matrix = pair_confusion_matrix(labels_true, labels_pred)
    s11 = matrix[1, 1]
    s10 = matrix[0, 1]
    s01 = matrix[1, 0]
    
    denominator = s11 + s10 + s01
    
    if denominator == 0:
        return 0.0
        
    return s11 / denominator