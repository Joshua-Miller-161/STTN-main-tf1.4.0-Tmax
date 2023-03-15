# @Time     : Mar. 14, 2023
# @Author   : Josh Miller / ChatGPT
# @FileName : distance_matrix.py
# @Version  : 1.0
# @IDE      : VSCode
# @Github   : ????


import math
import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the earth in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) * math.sin(dlat/2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2) * math.sin(dlon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c  # Distance in km
    return d

def distance_matrix(coords):
    n = len(coords)
    dist_matrix = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[j]
            dist = haversine(lon1, lat1, lon2, lat2)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return np.asarray(dist_matrix)