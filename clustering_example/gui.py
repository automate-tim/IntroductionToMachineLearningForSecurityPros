# First StreamLit App
import streamlit as st
# Vectorize_secrepo.py Libraries
import numpy as np
import os
import re
import h5py
import socket
import struct
from sklearn.preprocessing import normalize
# Visualize_vectors.py imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
# Cluster imports
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter


LOG_REGEX = re.compile(r'([^\s]+)\s[^\s]+\s[^\s]+\s\[[^\]]+\]\s"([^\s]*)\s[^"]*"\s([0-9]+)')


def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]


def get_prevectors():
    data_path = "data/www.secrepo.com/self.logs/"
    # ensure we get the IPs used in the examples
    prevectors = {
        ip2int("192.187.126.162"): {"requests": {}, "responses": {}},
        ip2int("49.50.76.8"): {"requests": {}, "responses": {}},
        ip2int("70.32.104.50"): {"requests": {}, "responses": {}},
    }
    for path in os.listdir(data_path):
        full_path = os.path.join(data_path, path)
        with open(full_path, "r") as f:
            for line in f:
                try:
                    ip, request_type, response_code = LOG_REGEX.findall(line)[0]
                    ip = ip2int(ip)
                except IndexError:
                    continue

                if ip not in prevectors:
                    if len(prevectors) >= 10000:
                        continue
                    prevectors[ip] = {"requests": {}, "responses": {}}

                if request_type not in prevectors[ip]["requests"]:
                    prevectors[ip]['requests'][request_type] = 0

                prevectors[ip]['requests'][request_type] += 1

                if response_code not in prevectors[ip]["responses"]:
                    prevectors[ip]["responses"][response_code] = 0

                prevectors[ip]["responses"][response_code] += 1

    return prevectors


def convert_prevectors_to_vectors(prevectors):
    request_types = [
        "GET",
        "POST",
        "HEAD",
        "OPTIONS",
        "PUT",
        "TRACE"
    ]
    response_codes = [
        200,
        404,
        403,
        304,
        301,
        206,
        418,
        416,
        403,
        405,
        503,
        500,
    ]

    vectors = np.zeros((len(prevectors.keys()), len(request_types) + len(response_codes)), dtype=np.float32)
    ips = []

    for index, (k, v) in enumerate(prevectors.items()):
        ips.append(k)
        for ri, r in enumerate(request_types):
            if r in v["requests"]:
                vectors[index, ri] = v["requests"][r]
        for ri, r in enumerate(response_codes):
            if r in v["responses"]:
                vectors[index, len(request_types) + ri] = v["requests"][r]

    return ips, vectors

def vectorize_data():
    prevectors = get_prevectors()
    ips, vectors = convert_prevectors_to_vectors(prevectors)
    vectors = normalize(vectors)

    with h5py.File("secrepo.h5", "w") as f:
        f.create_dataset("vectors", shape=vectors.shape, data=vectors)
        f.create_dataset("cluster", shape=(vectors.shape[0],), data=np.zeros((vectors.shape[0],), dtype=np.int32))
        f.create_dataset("notes", shape=(vectors.shape[0],), data=np.array(ips))

    print("Finished prebuilding samples")

# Visualize Funcs
def visualize(vectors):
    pca = PCA(n_components=3)
    projected_vectors = pca.fit_transform(vectors)
    # print(projected_vectors.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(
        projected_vectors[:, 0],
        projected_vectors[:, 1],
        zs=projected_vectors[:, 2],
        s=200,
    )
    st.pyplot(fig)
    # plt.show()

def general_clustering_functionality(clusters, vectors, ips):
    counter = Counter(clusters.tolist())
    for key in sorted(counter.keys()):
        st.write("Label {0} has {1} samples".format(key, counter[key]))

    # create new hdf5 with clusters added
    with h5py.File("secrepo.h5", "w") as f:
        f.create_dataset("vectors", shape=vectors.shape, data=vectors)
        f.create_dataset("cluster", shape=(vectors.shape[0],), data=clusters, dtype=np.int32)
        f.create_dataset("notes", shape=(vectors.shape[0],), data=np.array(ips))

def kmeans_clustering(vectors, number_clusters):
    kmeans = KMeans(n_clusters=number_clusters)
    clusters = kmeans.fit_predict(vectors)
    return clusters
    #general_clustering_functionality(clusters)

def dbscan_clustering(vectors, epsilon, number_points):
    dbscan = DBSCAN(eps=epsilon, min_samples=number_points)
    clusters = dbscan.fit_predict(vectors)
    return clusters
    # general_clustering_functionality(clusters)

#if __name__ == "__main__":
    # This takes the data from the data section of the repository and prepares it in vector format
    #vectorize_data()
    # Let's visualize the new vectors
    #path = "" # This path needs to be a file "h"
    #with h5py.File("secrepo.h5", "r") as f:
    #    vectors = f["vectors"][:]
    #visualize(vectors)

st.title('GUI for Intro to Machine Learning for Cyber Professionals')
with h5py.File("secrepo.h5", "r") as f:
        vectors = f["vectors"][:]
        ips = f["notes"][:]
if(st.button("Vectorize Data")):
    vectorize_data()
    st.write("Data has been vectorized!")
if(st.button("Visualize Data")):
        visualize(vectors)
# For Kmeans clustering, python cluster_vectors.py -c kmeans -n 2 -i secrepo.h5 -o secrepo.h5
if(st.button("K-Means Clustering")):
    st.write("Pick your hyper parameters")
    clusters = st.slider('Number of Clusters', 0, 20)
    if(st.button("Submit hyper parameter values")):
        new_clusters = kmeans_clustering(vectors, clusters)
        general_clustering_functionality(new_clusters, vectors, ips)
        st.success("Data modified with %d clusters", clusters)
# DBSCAN Clustering
if(st.button("DBSCAN Clustering")):
    st.write("Pick your hyper parameters")
    # Radius
    epsilon = st.slider('Epsilon', 0, 20, 2)
    min_samples = st.slider('Number of points', 0, 25, 5)
    if(st.button("Submit hyper parameter values")):
        new_clusters = dbscan_clustering(vectors, epsilon, min_samples)
        general_clustering_functionality(new_clusters, vectors, ips)
# Validate Clusters Statistically with Silhouette Scoring
if(st.button("Validate Clusters with Silhouette Scoring")):
    st.write("Validating Clusters")
# python stats_vectors.py secrepo.h5
