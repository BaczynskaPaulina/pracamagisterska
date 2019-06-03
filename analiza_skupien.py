from scipy.spatial.distance import hamming
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import jaccard_similarity_score
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle

#affinity propagation algorithm: [27][33]

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
    plt.show()
    
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
        plt.show()
        
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
            np_img=np_img.reshape(-1)
            if i in my_labels[0]:
                dataset2.append(np_img)
            if i in my_labels[1]:
                dataset3.append(np_img)
        dataset.append(dataset2)
        dataset.append(dataset3)
        
    return dataset

def get_information(vec1,vec2):
    d11=0
    d00=0
    d10=0
    d01=0
    for i in range(60000):
        if vec1[i]==1 and vec2[i]==1:
            d11=d11+1
        if vec1[i]==0 and vec2[i]==0:
            d00=d00+1
        if vec1[i]==1 and vec2[i]==0:
            d10=d10+1
        if vec1[i]==0 and vec2[i]==1:
            d01=d01+1
    return d11,d00,d10,d01

def do_hamming(m,hmm,vec):
    for i in range(m):
        for j in range(m):
            hmm[i,j]=hamming(vec[i],vec[j])
    return hmm

def do_vari(m,vari,vec):
    for i in range(m):
        for j in range(m):
            d11,d00,d10,d01=get_information(vec[i],vec[j])
            vari[i,j]=(d01+d10)/(4*(d11+d01+d10+d00))
    return vari

def do_roger_tanimoto(m,rogtan,vec):
    for i in range(m):
        for j in range(m):
            rogtan[i,j]=distance.rogerstanimoto(vec[i],vec[j])
    return rogtan

def do_size_difference(m,sd,vec):
    for i in range(m):
        for j in range(m):
            d11,d00,d10,d01=get_information(vec[i],vec[j])
            a=d01+d10
            b=d11+d01+d10+d00
            sd[i,j]=pow(a,2)/pow(b,2)
    return sd
            
def do_pattern_difference(m,pdif,vec):
    for i in range(m):
        for j in range(m):
            d11,d00,d10,d01=get_information(vec[i],vec[j])
            b=d11+d01+d10+d00
            pdif[i,j]=(4*d01*d10)/pow(b,2)
    return pdif

def do_jaccart_needham(m,jcc,vec):
    for i in range(m):
        for j in range(m):
            jcc[i,j]=distance.jaccard(vec[i],vec[j])
    return jcc

def do_sokal_sneath(m,sok,vec):
    for i in range(m):
        for j in range(m):
            sok[i,j]=distance.sokalsneath(vec[i],vec[j])
    return sok

def do_russel_rao(m,rus,vec):
    for i in range(m):
        for j in range(m):
            rus[i,j]=distance.russellrao(vec[i],vec[j])
    return rus

def do_sokal_sneath_4(m,sok4,vec):
    for i in range(m):
        for j in range(m):
            d11,d00,d10,d01=get_information(vec[i],vec[j])
            a1=d11/(d11+d01)
            a2=d11/(d11+d10)
            a3=d00/(d01+d00)
            a4=d00/(d10+d00)
            li=a1+a2+a3+a4
            sok4[i,j]=li/4
    return sok4

def do_sokal_sneath_5(m,sok5,vec):
    for i in range(m):
        for j in range(m):
            d11,d00,d10,d01=get_information(vec[i],vec[j])
            a1=d11+d01
            a2=d11+d10
            a3=d01+d00
            a4=d10+d00
            p=a1*a2*a3*a4
            sok5[i,j]=(d11*d00)/np.sqrt(p)
    return sok5

def do_yule(m,yule,vec):
    for i in range(m):
        for j in range(m):
            yule[i,j]=distance.yule(vec[i],vec[j])
    return yule

def do_hellinger(m,hel,vec):
    for i in range(m):
        for j in range(m):
            d11,d00,d10,d01=get_information(vec[i],vec[j])
            a1=d11+d01
            a2=d11+d10 
            a=a1*a2
            b=d11/(np.sqrt(a))
            hel[i,j]=2*np.sqrt(1-b)
    return hel

def do_gower(m,gow,vec):
    for i in range(m):
        for j in range(m):
            d11,d00,d10,d01=get_information(vec[i],vec[j])
            a1=d11+d01
            a2=d11+d10
            a3=d01+d00
            a4=d10+d00
            p=a1*a2*a3*a4
            gow[i,j]=(d11+d00)/np.sqrt(p)
    return gow

def do_goodman_kruskal(m,gk,vec):
    for i in range(m):
        for j in range(m):
            d11,d00,d10,d01=get_information(vec[i],vec[j])
            
            a1=max(d11,d01)
            a2=max(d10,d00)
            a3=max(d11,d10)
            a4=max(d01,d00)
            
            a5=max(d11+d10,d01+d00)
            a6=max(d11+d01,d10+d00)
            
            s1=a1+a2+a3+a4
            s2=a5+a6

            gk[i,j]=(s1-s2)/((2*m)-s2)
    return gk

def do_single_linkage(m,pom,title):
    Z=linkage(pom,'single')
    fig=plt.figure(figsize=(25, 10))
    dn=dendrogram(Z)
    t="Single - linkage: "+title
    plt.title(t)
    plt.show()
    
def do_complete_linkage(m,pom,title):
    Z=linkage(pom,'complete')
    fig=plt.figure(figsize=(25, 10))
    dn=dendrogram(Z)
    t="Complete - linkage: "+title
    plt.title(t)
    plt.show()
    
def do_median_linkage(m,pom,title):
    Z=linkage(pom,'median')
    fig=plt.figure(figsize=(25, 10))
    dn=dendrogram(Z)
    t="Median - linkage: "+title
    plt.title(t)
    plt.show()
    
def do_affinity_prop(m,pom,title):
    af=AffinityPropagation().fit(pom)
    print(af.labels_)
    cluster_centers_indices=af.cluster_centers_indices_
    labels=af.labels_
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
    
def do_clustering(m,pom,title):
    do_single_linkage(m,pom,title)
    do_complete_linkage(m,pom,title)
    do_median_linkage(m,pom,title)
    do_affinity_prop(m,pom,title)
    
def write(writer,pom,t):
    df=pd.DataFrame(data=pom)
    df.to_excel(writer, sheet_name=t)
            
def do_analysis(dataset):
    writer=pd.ExcelWriter('Results.xlsx', engine='xlsxwriter')
    for i in range(len(dataset)):
        m=len(dataset[i])
        vec=dataset[i]
        pom=np.empty(shape=(m,m))
        
        t=str(i)+". Hamming Distance"
        do_hamming(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Vari Distance"
        do_vari(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Roger&Tan. Dissim."
        do_roger_tanimoto(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Size-Diff. Sim."
        do_size_difference(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Pattern-Diff. Sim."
        do_pattern_difference(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Jaccard-N. Dissim."
        do_jaccart_needham(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Sokal&Sneath Diss."
        do_sokal_sneath(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Russer&Rao Sim."
        do_russel_rao(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Sokal&Sneath Sim.(IV)"
        do_sokal_sneath_4(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Sokal&Sneath Sim.(V)"
        do_sokal_sneath_5(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Yule Dissim."
        do_yule(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Hellinger Dist."
        do_hellinger(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Gower Sim."
        do_gower(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        t=str(i)+". Goodman&Kr. Sim."
        do_goodman_kruskal(m,pom,vec)
        write(writer,pom,t)
        do_clustering(m,pom,t)
        
        
if __name__ == "__main__":
    sizes=import_files_and_get_sizes()
    print_sizes(sizes)
    labels=do_kmeans_if_needed(sizes)
    normalize()
    dataset=change_to_binary(labels)
    do_analysis(dataset)
    