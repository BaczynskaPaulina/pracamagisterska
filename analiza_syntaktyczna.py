import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from scipy.spatial.distance import hamming
from PIL import Image, ImageOps
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from itertools import cycle

#affinity propagation algorithm: [25][31]

n=38

def import_files_and_get_sizes():
    sizes=np.empty(shape=(n,2))

    for i in range(n):
        image_file=str(i)+'.jpg'
        im=Image.open("dataset-original-size/"+image_file)
        width, height = im.size
        sizes[i][0]=im.size[0]
        sizes[i][1]=im.size[1]
    return sizes

def print_sizes(sizes):
    columns = ['x', 'y']
    df = pd.DataFrame(data=sizes,columns=columns)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(df['x'], df['y'], color='k')
    plt.xlim(100, 3000)
    plt.ylim(100, 3000)
    plt.title("SIZES")
    #plt.show()
    
def mul(sizes):
    mulled_sizes=np.empty(shape=(n))
    for i in range(n):
        mulled_sizes[i]=sizes[i][0]*sizes[i][1]
    return mulled_sizes

def check_sizes(mulled_sizes):
    for i in range(n):
        for j in range(n):
            if mulled_sizes[i]/mulled_sizes[j]>3:
                return True
    return False

def do_kmeans_if_needed(sizes):
    mulled_sizes=mul(sizes)
    if check_sizes(mulled_sizes):
        columns=['x','y']
        df=pd.DataFrame(data=sizes,columns=columns)
        kmeans=KMeans(n_clusters=2)
        kmeans.fit(df)
        labels = kmeans.predict(df)
        centroids = kmeans.cluster_centers_
        fig = plt.figure(figsize=(10, 10))
        colors = map(lambda x: colmap[x+1], labels)
        colmap = {1: 'red', 2: 'yellow', 3: 'green', 4: 'purple', 5: 'blue'}
        plt.scatter(df['x'], df['y'], c=kmeans.labels_, alpha=0.5, edgecolor='k')
        for idx, centroid in enumerate(centroids):
            plt.scatter(*centroid, color=colmap[idx+1])
        plt.xlim(100, 3000)
        plt.ylim(100, 3000)
        plt.title("AFTER KMEANS CLUSTERING")
        #plt.show()
        
        my_labels=[]
        for i in range(kmeans.n_clusters):
            my_labels.append([])

        for i in range(kmeans.n_clusters):
            for j in range(n):
                if kmeans.labels_[j]==i:
                    my_labels[i].append(j)
        return my_labels
    else:
        my_labels=[]
        my_labels.append(list(range(n)))
        return my_labels
    
def normalize():
    for i in range(n):
        image_file=str(i)+'.jpg'
        im=Image.open("dataset-original-size/"+image_file)
        im2=im.resize((200, 300),Image.NEAREST)
        im2.save("dataset-resize/"+image_file)
        
def change_to_binary(my_labels):
    dataset=[]
    dataset2=[]
    dataset3=[]
    if len(my_labels)==1:
        for i in range(n):
            image_file=str(i)+'.jpg'
            img=Image.open("dataset-resize/"+image_file).convert('L')
            img_inverted=ImageOps.invert(img)
            np_img = np.array(img_inverted)
            np_img[np_img>0]=1
            np_img=np_img.reshape(-1)
            dataset2.append(np_img)
        dataset.append(dataset2)
            
    if len(my_labels)==2:
        for i in range(n):
            image_file=str(i)+'.jpg'
            img=Image.open("dataset-resize/"+image_file).convert('L')
            img_inverted=ImageOps.invert(img)
            np_img = np.array(img_inverted)
            np_img[np_img>0]=1
            if i in my_labels[0]:
                dataset2.append(np_img)
            if i in my_labels[1]:
                dataset3.append(np_img)
        dataset.append(dataset2)
        dataset.append(dataset3)
        
    return dataset

def get_mean(dataset):
    l=len(dataset)
    list_of_mean_dataset=[]
    for a in range(l):
        list_of_means=[]
        b=len(dataset[a])
        for i in range(b):
            means=np.zeros(shape=(300))
            for j in range(300):
                listOf1row=[]
                for k in range(200):
                    if dataset[a][i][j][k]==1:
                        listOf1row.append(k) 
                if(len(listOf1row)==0):
                    means[j]=0
                else:
                    mini=min(listOf1row)
                    maxi=max(listOf1row)
                    mean=int((mini+maxi)/2)
                    means[j]=mean
            list_of_means.append(means)
        list_of_mean_dataset.append(list_of_means)
    return list_of_mean_dataset

def divide_into_f(list_of_mean_dataset):
    result=[]
    for i in range(len(list_of_mean_dataset)):
        m=len(list_of_mean_dataset[i])
        result2=np.zeros(shape=(m,50))
        for k in range(m):
            a=0
            for j in range(0,293,6):
                result2[k][a]=(list_of_mean_dataset[i][k][j]-list_of_mean_dataset[i][k][j+5])
                a=a+1
        result.append(result2)
    return result

def do_euclidean(m,euc,vec):
    for i in range(m):
        for j in range(m):
            euc[i,j]=distance.euclidean(vec[i],vec[j])
    return euc

def do_affinity_prop(m,pom,title):
    af=AffinityPropagation().fit(pom)
    cluster_centers_indices=af.cluster_centers_indices_
    labels=af.labels_
    print(af.labels_)
    n_clusters_ = len(cluster_centers_indices)
    plt.close('all')
    plt.figure(1)
    plt.clf()
    colors=cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_),colors):
        class_members=labels==k
        cluster_center=pom[cluster_centers_indices[k]]
        plt.plot(pom[class_members, 0],pom[class_members, 1],col + '.')
        plt.plot(cluster_center[0], cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k', markersize=14)
        for x in pom[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    t="Affinity Propagation: "+title
    plt.title(title)
    plt.show()
    
def do_single_linkage(m,pom,title):
    Z=linkage(pom,'single')
    fig=plt.figure(figsize=(25, 10))
    dn=dendrogram(Z)
    t="Single - linkage: "+title
    plt.title(t)
    plt.show()
    
def write(writer,pom,t):
    df=pd.DataFrame(data=pom)
    df.to_excel(writer, sheet_name=t)
    
if __name__ == "__main__":
    sizes=import_files_and_get_sizes()
    print_sizes(sizes)
    labels=do_kmeans_if_needed(sizes)
    normalize()
    dataset=change_to_binary(labels)
    means=get_mean(dataset)
    result=divide_into_f(means)
    
    writer=pd.ExcelWriter('Results2.xlsx', engine='xlsxwriter')
    for i in range(len(result)):
        m=len(result[i])
        vec=result[i]
        pom=np.empty(shape=(m,m))
        t=str(i)+". Euclidean Distance"
        do_euclidean(m,pom,vec)
        write(writer,pom,t)
        do_affinity_prop(m,pom,t)
        