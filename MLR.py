import numpy as np
from scipy.sparse import csr_matrix
import time
import os
import cPickle as pickle
import matplotlib.pyplot as plt

class MLR():
    """
        Perform Multinomial Logistic Regression, also called softmax
        regression for the task of predicting the rating based on text review.
        The algorithm uses batched stochastic gradient update method. The 
        stopping criteria is defined by number of epochs, where each epoch is a
        pass over the entire training dataset.   
    """
    def __init__(self,data,label,lr,lamda,max_epoch):
        """
            Initialize the model
            
            Parameters
            ---------
            data: csr_matrix
                sparse matrix representation for training data
            label: ndarray
                dense  matrix with each row as one hot vector representing the 
                rating value
            lr: float
                learning rate
            lamda: float
                regularization factor
            max_epoch: int
                maximum number of iterations to run, stopping criteria
                
            Note
            -----
            creates several private variables corresponding to the input
            parameters. Also initializes the weights matrix, theta.
        """
        self._lr = lr #learning rate
        self._lamda = lamda #regularization factor
        self._max_epoch = max_epoch #stopping criteria
        self._data = data #training data
        self._label = label #training labels
        [self._N,self._M] = self._data.get_shape() #number of data points in training and their dimension
        self._C = self._label.shape[1] #number of classes
        self._theta  = np.random.normal(size = (self._M,self._C)) #1000x5 weights matrix

        
    def predict(self,x):
        """
            Carries out the the prediction of the rating based on the weights.
            
            Parameters
            ----------
            x: csr_matrix
                sparse matrix of data
            
            Returns
            --------
            p: ndarray
                dense matrix containing probability of each class for each data
                point.
        """
        thetaX = x.dot(self._theta) #wx
        thetaX_exp = np.exp(thetaX) #exp(wx)
        thetaX_exp_sum = np.sum(thetaX_exp,axis=1) #sum of exp(wx)
        p = thetaX_exp.T/thetaX_exp_sum #5xlen(x) predicted results
        
        if np.isinf(p).any(): #deal with overflow in results.
            inf_idx = np.isinf(p) #idx where overflow occurs
            val = np.sum(p,0)/np.sum(inf_idx,0)*inf_idx #values to be used to substitution
            p[inf_idx]=val[inf_idx]#substitute values
        
        return p.T 


    def update(self,y,x):
        """
            Carry out the gradient update. The update is split in two parts, 
            regularization update and empirical update.
            
            Parameters
            ----------
            y: ndarray
                labels of each data point in x
            x: csr_matrix
                sparse matrix containing one batch of data
            
            Returns
            -------
            cost: the cost associated with the model
        """
        [batch_size,_] = x.get_shape() #get batch_size
        p = self.predict(x) #find prediction result for x
        emp_update = x.T.dot((y-p))*self._lr/batch_size # calculate empirical update value
        reg_update = (1-self._lamda*self._lr) #calculate regularization update value
        self._theta = reg_update*self._theta #apply regularization update
        self._theta += emp_update#apply empirical update
        
        #calculate the cost associated.
        cost = (-1.0/batch_size)*np.sum(y*np.log(p)) +  (self._lamda/2*np.sum(np.power(self._theta,2)))
        return cost
    
    
    def reinit(self):
        """
            Shuffles the training data
        """
        state = np.random.get_state() #get random state
        np.random.shuffle(self._data.todense()) #shuffle data
        self._data = csr_matrix(self._data) 
        np.random.set_state(state) #reset the state
        np.random.shuffle(self._label) #shuffle label

    def train(self,batch_size,figurepath):
        """
            Training the MLR model based on the parameter values available
            
            Parameters
            -----------
            batch_size: int
                mini-batch size
            figurepath: string
                path where the plot of cost with epochs is stored
            
        """
        epoch_num = 1 #current epoch number
        cost_list = list() #list to store average cost per epoch
        
        #for each epoch
        while(epoch_num < self._max_epoch):
            batch_num = 1 
            st = time.time()
            batch_end = 0
            cost = 0
#             self.reinit()#shuffle the data
            #for each batch of data
            while(batch_end<self._N):
                batch_start = batch_size*(batch_num-1) 
                batch_end = np.min((batch_size*batch_num, self._N)) 
            
                x= self._data[batch_start:batch_end,:] #get batch of data
                y=self._label[batch_start:batch_end,:] #get corresponding batch of labels
                
                batch_num += 1
        
                cost += self.update(y, x) #carry out gradient update for the batch
            cost_list.append(cost/batch_num) #store average cost per iteration
            # print "Epoch Number:",epoch_num,"Cost:",cost/batch_num, "Time taken:",time.time()-st
            
            epoch_num +=1
        plt.plot(np.arange(len(cost_list)),cost_list) #plot the cost 
        # plt.title("Average Cost per Epoch during Training Phrase")
        plt.ylabel("Cost")
        plt.xlabel("Epoch number")

        # plt.imshow() #display the cost over the epochs
        plt.savefig(figurepath)
        plt.close()

    def test(self,data):
        """
            Test the data based on the trained model
            
            Parameters
            -----------
            data: csr_matrix
                msparse matrix containing the input data
            
            Returns
            --------
            hard_prediction: ndarray
                Hard prediction results. each value is a rating
            soft_prediction: ndarray 
                Soft prediciton results. Equals to sum of ratings and the
                associated probability.
        """
        #return hard and soft score
        p = self.predict(self._data) #predict results
        
        #Verify for training data
        hard_prediction = np.argmax(p,1)+1 #hard prediction
        true_label = np.argmax(self._label,1)+1 #true labels
        
        print "Training Accuracy ",float(np.sum(hard_prediction==true_label))/self._N
        print "Training RMSE ",np.sqrt(float(np.sum(np.power(hard_prediction-true_label,2)))/self._N)
        
        # for test data
        p = self.predict(data) #predict results
        hard_prediction = np.argmax(p,1)+1 #hard prediction
        soft_predicton = p.dot((np.arange(self._C)+1)) #soft prediction
        
        return hard_prediction,soft_predicton
        
def main():
    basepath = "../output"
    name = "train"

    labelfile = "{}/task1/{}/{}.rating".format(basepath,name,name)
    label = pickle.load(open(labelfile,"r"))
    
    for feature_type in [ "base_feature","base_tfidf",
        "base_norm","hash_feature","hash_tfidf","hash_norm",
        "hash_feature_stem","hash_tfidf_stem","hash_norm_stem",
        "base_feature_stem","base_tfidf_stem","base_norm_stem"]:
        
        name = "train"
        print feature_type
        
        datafile = "{}/task2/{}/{}.{}".format(basepath,name,name,feature_type)
        
        #read the data
        data = pickle.load(open(datafile,"r"))
        print "data read"
        #train model
        figurepath = "../images/{}.png".format(feature_type)
        if not os.path.exists(os.path.dirname(figurepath)):
            os.makedirs(os.path.dirname(figurepath))
        
        mlr = MLR(data,label,lr=0.05,lamda=0.0005,max_epoch=100)
        mlr.train(batch_size=1000, figurepath=figurepath)
        
        name = "dev"
        datafile = "{}/task2/{}/{}.{}".format(basepath,name,name,feature_type)
        outfile = "{}/task3/{}.{}".format(basepath,name, feature_type)
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        
        #load data
        data = pickle.load(open(datafile,"r"))
        #test
        [hard_pred, soft_pred]=mlr.test(data)
         
        #write results to file
        with open(outfile,"w+") as fp:
            for h,s in zip(hard_pred,soft_pred):
                fp.write("{} {}\n".format(h,s))
        
    
if __name__ == "__main__":
    main()