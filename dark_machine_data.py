import numpy as np
import pickle
import matplotlib.pyplot as plt
from utilities import standardize


def line_to_features_MET_Pobj(line, max_obj_num):
    a=line.split(";")
    a.pop()
    x=np.zeros(2+max_obj_num*4)
    x[0]=float(a[3])
    x[1]=float(a[4])
    index=2
    for i in range(5, len(a)):
        b=a[i].split(",")
        #print(b)
        for j in range(4):
            x[index]=float(b[j+1])
            index+=1
    return x


def prepare_data_dark_machines(path, crop=None, field=None, standard=None): 
    text_file = open(path)
    max_obj_num=20
    X=[]
    if field==None:
        for line in text_file:
            X.append(line_to_features_MET_Pobj(line, max_obj_num))
    else:
        if field[0]==None:
            f0=0
        else:
            f0=field[0]
        
        if field[1]==None:
            f1=np.inf
        else:
            f1=field[1]
        i=0
        for line in text_file:
            if i<f1 and i>=f0:
                X.append(line_to_features_MET_Pobj(line, max_obj_num))
            i+=1
        
    if crop!=None:
        if len(X)<crop:
            print("dataset is too short!")
    X=np.array(X)
    standard=standardize(X, standard=standard)
    return X, standard

if __name__ == "__main__":
    X, standard = prepare_data_dark_machines("C:/bachelor work/Spyder/data_set/training_files/chan2a/background_chan2a_309.6.csv", crop=None, field=None, standard=None)
    
    