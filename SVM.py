import sys
sys.path.append("./liblinear-2.1/python")
import os
from liblinear import *
from liblinearutil import *
import pickle
import time
import numpy as np

class MSVM():
    """
        Perform multiclass L2-regularized L2-loss Support Vector Classification
        using Liblinear library available at https://www.csie.ntu.edu.tw/~cjlin/liblinear/
        The functions included in this class include a method to format data in
        required format for the Liblinear package.
        Additionally, a wrapper for training the model is also present.
    """
    def format_data(self,data,label):
        """
        
        """
        x=list()
        y=list()
        for i in range(data.get_shape()[0]):
            temp = dict()
            for c,d in zip(data.getrow(i).indices,data.getrow(i).data):
                temp[c]=d
            x.append(temp)
            if len(label)>0:
                y.append(int(np.argmax(label[i])+1))
        return [y,x]
    
    def train(self,y,x):
        data = problem(y,x)
        param = parameter("-s 1 -v 10")
        model_ptr = liblinear.train(data, param) 
        model_ = toPyModel(model_ptr)
        return model_
    
    
def main():
    st = time.time()
    basepath = "/Users/karishma/Dropbox/CMU/fall_2015/MLT/hw5/resources"
    for feature_type in ["base_feature","base_tfidf","base_norm",
                         "hash_feature","hash_tfidf","hash_norm",
                         "base_feature_stem","base_tfidf_stem","base_norm_stem",
                         "hash_feature_stem","hash_tfidf_stem","hash_norm_stem"]:
    
        name = "train"
        datafile = "{}/task2/{}/{}.{}".format(basepath,name,name,feature_type)
        labelfile = "{}/task1/{}/{}.rating".format(basepath,name,name)
        
        label=pickle.load(open(labelfile,"r"))
        data = pickle.load(open(datafile,"r"))
        
        
        svm=MSVM() #create model
        [y,x]=svm.format_data(data, label) #format input data in required format
        
        model = svm.train(y, x) #train model
        
        p_labs, p_acc, p_vals = predict(y, x, model) #training accuracy
        
        name = "dev"
        datafile = "{}/task2/{}/{}.{}".format(basepath,name,name,feature_type)
        outfile = "{}/task4/{}/{}_result.{}".format(basepath,name,name,feature_type)
        data = pickle.load(open(datafile,"r"))
        
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        
        [y,x]=svm.format_data(data,[])#format data
        
        p_labs, p_acc, p_vals = predict(y, x, model) #testing results
        
        writer = open(outfile,"w+")#write test labels
        for i in range(len(p_labs)):
            writer.write("{} 0.0\n".format(int(p_labs[i])))
        writer.close()
        
        print feature_type, time.time()-st
        
    
if __name__ == "__main__":
    main()
    