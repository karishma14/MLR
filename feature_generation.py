'''
Created on Nov 18, 2015

@author: karishma
'''

import numpy as np
import time
import os
from scipy.sparse import csr_matrix
from _heapq import heappush, heapreplace, heappop
import cPickle as pickle
from helper import norm


class feat_gen:
    """
         This class should take the pre-processed tokens for each review
         and generate a sparse matrix based on either the baseline model
         or hash model.
         
         The baseline model simply considers dictionary of most frequent k 
         words. Each review is then mapped to these k words and the rest of 
         the words are ignored
         
         The hash model takes two parameters. Vocabulary size v and 
         dictionary/hash size k. Here the most frequent v words are taken and 
         hashed to a dictionary of size k. We expect due to sparsity of tokens 
         in text there would not be too many collisions while hashing and the
         few which do occur can be safely ignored.
         
         In both of these cases, we were using the term frequency directly as 
         attribute value. Instead of that, TF-IDF can be considered or explicit
         normalizing between [0-1] can be done. This would also help with the 
         issue overflow during the gradient update calculation for logistic 
         regression. 
         
    """
    
    def _dictionary(self, filename, k):
        """
            This function reads the file which contains the token list and
            their frequency in the entire training corpus. Then it selects the
            top k tokens and generates the dictionary.
            
            Parameters
            -----------
            filename: string
                Path of the file which contains the tokens and their frequencies
            k: int
                dictionary size
            
            Note
            -----
            this method creates an private variable self._vocab which 
            contains the frequent tokens and an unique id associated with them
            and self._idf which stores the term frequency of each token in self._vocab
        """
        
        temp_vocab = []
        

        data = pickle.load(open(filename, "r"))
        for line in data.keys():
            # if the size of heap is smaller than k then add element
            if len(temp_vocab) <= k:
                heappush(temp_vocab, (int(data[line]), line))
            else:
            #if the size of heap is greater than k then replace smallest element
                heapreplace(temp_vocab, (int(data[line]), line))
                
        heappop(temp_vocab)  # remove the extra element
        
        self._vocab = dict()
        itr = 0
        self._idf = list()
        
        for term in list(temp_vocab):
            #convert heap into dictionary keyed on the token and value is token id
            self._vocab[term[1]] = itr
            #store the term frequency in case of tfidf calculations
            self._idf.append(term[0])
            itr += 1
        self._idf = np.array(self._idf)

    
    def basic_feat(self, reviewsfile, dictfile, outputfile, tfidf_file, 
                   norm_file, k, normFlag=False, tfidf=False):
        """
            This method simply maps each review text to sparse feature matrix.
            The number of features is defined by dictionary size. The value of
            each cell in the sparse matrix is just the number of times that
            token occurred in the review.
            
            Parameters
            -----------
            reviewsfile: string
                path of the file containing the reviews. each line is a new review
            dictfile: string
                path of file containing the tokens in training corpus and their 
                frequencies.
            outfile: string
                path of the output file
            tfidf_file: string
                path of output file to store tfidf features
            norm_file: string
                path of output file to store normalized features
            k: int
                size of the dictionary
            normFlag: bool
                when true perform standard normalization, rescale the data 
                to [0,1]
            tfidf: bool
                when true replace term frequency with tf-idf
                
            Note
            -----
            Output file format is sparse matrix
        """
        
        self._dictionary(dictfile, k) #create the dictionary of top k terms
        row = list()
        column = list()
        val = list() #base values
        val_idf = list() #tfidf version of base values
        val_norm = list() #norm version of base values
        line_no = 0
        
        # for each line
        for line in pickle.load(open(reviewsfile, "r")):
            #calculate term frequency for the review text wrt dictionary
            #this is simply word count
            tf = np.zeros(len(self._vocab))
            for term in line.split(" "):
                if self._vocab.has_key(term):
                    tf[self._vocab[term]] += 1
            #non zero term index
            non_zero = tf > 0
            
            data = tf[non_zero] #non zero terms
            col = np.arange(len(self._vocab))
            col = col[non_zero]
            
            #create a list for use in sparse matrix initialization later
            for c, d in zip(col, data):
                row.append(line_no)
                column.append(c)
                val.append(d)
                
            #calculate the tfidf version for non zero terms
            if tfidf == True:
                data_tfidf = tf[non_zero] / self._idf[non_zero]
                [val_idf.append(d) for d in data_tfidf]
            
            #calculate the normalized version for non zero terms
            if normFlag == True:
                data_norm = norm(tf[non_zero])
                [val_norm.append(d) for d in data_norm]
            
            line_no += 1
        #create sparse matrix and dump to output file
        mat = csr_matrix((val, (row, column)), (line_no, k))
        pickle.dump(obj=mat, file=open(outputfile, "w+"))
        print "base features generated"
        
        #create sparse matrix of tfidf terms and dump to output
        if tfidf == True:
            mat = csr_matrix((val_idf, (row, column)), (line_no, k))
            pickle.dump(obj=mat, file=open(tfidf_file, "w+"))
            print "base features with tfidf generated"
        
        #create sparse matrix of normalized terms and dump to output
        if normFlag == True:
            mat = csr_matrix((val_norm, (row, column)), (line_no, k))
            pickle.dump(obj=mat, file=open(norm_file, "w+"))
            print "base features with normalization generated"
        
        
    def hash_feat(self, reviewsfile, dictfile, outputfile, tfidf_file, norm_file, v, k, normFlag=False, tfidf=False):
        """
            This method first generates the dictionary of specified size.
            It then considers only the terms from dictionary in each review
            and maps it to a feature space by hashing
            Simply to the previous case the output is simply a sparse feature
            matrix where each token is term frequency wrt the review text.
            
            Parameters
            ----------
            reviewsfile: string
                path of the file containing the reviews. each line is a new review
            dictfile: string
                path of file containing the tokens in training corpus and their 
                frequencies.
            outfile: string
                path of the output file
            tfidf_file: string
                path of output file to store tfidf features
            norm_file: string
                path of output file to store normalized features
            v: int 
                size of vocabulary
            k: int
                size of the dictionary
            normFlag: bool
                when true perform standard normalization, rescale the data 
                to [0,1]
            tfidf: bool
                when true replace term frequency with tf-idf
        """

        self._dictionary(filename=dictfile, k=v) #create the dictionary of top k terms
        row = list()
        column = list()
        val = list()#base values
        val_norm = list()#norm version of base values
        val_idf = list()#tfidf version of base values
        line_no = 0
        
        for line in pickle.load(open(reviewsfile, "r")):
            #store term frequency wrt current review and overall
            tf = np.zeros(k)
            idf = np.zeros(k)
            for term in line.split(" "):
                if self._vocab.has_key(term):
                    tf[hash(term) % k] += 1
                    #in case of collisions store the highest tf
                    idf[hash(term) % k] = np.max((idf[hash(term) % k], self._idf[hash(term) % k])) 
        
            non_zero = tf > 0 #non zero term index
            data = tf[non_zero]
            col = np.arange(k)
            col = col[ non_zero]
            
            #create a list for use in sparse matrix initialization later
            for c, d in zip(col, data):
                row.append(line_no)
                column.append(c)
                val.append(d)
            
            #create sparse matrix of tfidf terms and dump to output
            if tfidf == True:
                data_tfidf = tf[non_zero] / idf[non_zero]
                [val_idf.append(d) for d in data_tfidf]
            
            #create sparse matrix of normalized terms and dump to output
            if normFlag == True:
                data_norm = norm(tf[non_zero])
                [val_norm.append(d) for d in data_norm]
            
            line_no += 1

        #create sparse matrix and dump to output file
        mat = csr_matrix((val, (row, column)), (line_no, k))
        pickle.dump(obj=mat, file=open(outputfile, "w+"))
        print "hash features generated"
        
        #create sparse matrix of tfidf terms and dump to output
        if tfidf == True:
            mat = csr_matrix((val_idf, (row, column)), (line_no, k))
            pickle.dump(obj=mat, file=open(tfidf_file, "w+"))
            print "hash features with tfidf generated"
        
        #create sparse matrix of normalized terms and dump to output
        if normFlag == True:
            mat = csr_matrix((val_norm, (row, column)), (line_no, k))
            pickle.dump(obj=mat, file=open(norm_file, "w+"))
            print "hash features with normalization generated"
        
def main():
    basepath = "../output"
    dictfile = "{}/task1/train/train.mapping_stem".format(basepath)
    v = 10000
    k = 1000
    
    for name in ["debug","dummy","test","dev","train"]:
        reviewsfile = "{}/task1/{}/{}.base".format(basepath, name, name)
        
        base_feature = "{}/task2/{}/{}.base_feature_stem".format(basepath, name, name)
        base_tfidf = "{}/task2/{}/{}.base_tfidf_stem".format(basepath, name, name)
        base_norm = "{}/task2/{}/{}.base_norm_stem".format(basepath, name, name)
        hash_feature = "{}/task2/{}/{}.hash_feature_stem".format(basepath, name, name)
        hash_tfidf = "{}/task2/{}/{}.hash_tfidf_stem".format(basepath, name, name)
        hash_norm = "{}/task2/{}/{}.hash_norm_stem".format(basepath, name, name)
        st = time.time()
    
        #create directory structure if needed
        if not os.path.exists(os.path.dirname(base_feature)):
            os.makedirs(os.path.dirname(base_feature))
        
        features = feat_gen()
        features.basic_feat(reviewsfile, dictfile, base_feature, base_tfidf, 
                            base_norm, k, normFlag=True, tfidf=True)
        
        print name, "basic features", time.time() - st
        st = time.time()
        
        features.hash_feat(reviewsfile, dictfile, hash_feature, hash_tfidf, 
                           hash_norm, v, k, normFlag=True, tfidf=True)
        
        print name, "hash features", time.time() - st
        st = time.time()
        
if __name__ == "__main__":
    main()
