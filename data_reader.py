'''
Created on Nov 3, 2015

@author: karishma
'''

import json
import re
import time
import os
import numpy as np
import cPickle as pickle
from nltk.stem.porter import PorterStemmer
from helper import onehot

class data_reader:
    """
        The data reader carried out basic data processing after reading it 
        from the json file. The default processing includes tokenization, 
        removal of stopwords, and removal of terms containing numerical values.
        
        Additionally, you can carry out stemming by explicitly setting the
        appropriate flag.
        
    """
    def __init__(self, stoplist):
        self.stopwords(stoplist)
        
    def stopwords(self, stoplist):
        """
        read all the stopwords and store it in a list, to be used later on for pre-processing
        
        Parameters
        -------------
        stoplist: String
            contains path of the file which contains the stopwords
        
        Note
        ------
        creates a private variable self._stopwords which contains all the stopwords 
        """
        # read all the stopwords in memory
        self._stopwords = dict()
        with open(stoplist) as fp:
            for line in fp.readlines():
                self._stopwords[line.strip()] = 1
                
    def read_json(self, inputfile, outfile_rate, outfile_token, stem=False):
        """
            read the data from the json file and extract useful fields. Here
            some basic preprocessing is carried out. The processing steps are
            applied in the following sequence.
            1. The review text is tokenized.
            2. Each token is converted to lowercase
            3. Punctuations or special characters are removed
            4. Tokens with numerical characters are removed
            5. Tokens are filtered against stopword list
            6. If stemming flag was set then the tokens are stemmed.
             
            
            Parameters
            ------------
            inputfile: String
                contains the path of the input json file
            outfile_rate: string
                path for storing the ratings if available.
            outfile_token: string
                path for storing the processed reviews. Each line is a string
                with string separated tokens
            stem: bool
                Flag to specify if stemming should be carried out or not
            
            
        """
    
        self._token_map = dict() #unique words which appear in text with their tf
        review = []   #list for reviews
        rating = []  #list for ratings
        stemmer = PorterStemmer() #stemmer from nltk
        reader = open(inputfile, "r")
        self._df = dict()
        
        for line in reader.readlines():
            temp = json.loads(line)
            tokens = temp['text'].split(" ") #split into tokens separated by space. Step 1
            value = "" #dummy string to store result
            flag = dict() # reinitialize the flag for each review, used for calculating term-document freq
            for token in tokens:
                token = token.lower() #Step 2.convert to lower case.
                token = re.sub('[^a-z0-9]+', '', token)#Step 3.remove special characters
                
                if token.isalpha(): #Step 4.consider only words with alphabets and no numericals
                
                    if self._stopwords.has_key(token):  #Step 5.if present in stopword list then ignore 
                        continue
                    else:
                    
                        if stem == True:
                            token = stemmer.stem(token) #Step 6. stem the token
                        value += "{} ".format(token)
                        
                        if self._token_map.has_key(token): #keep track of tf in dictionary
                            self._token_map[token] += 1
                            if not flag.has_key(token):
                                self._df[token] += 1
                                flag[token]=1
                        else:
                            self._token_map[token] = 1
                            self._df[token]=1
                            flag[token]=1
            review.append(value) #add processed review text to list
            
            if temp.has_key("stars"): #if there are ratings then store that
                rating.append(onehot(int(temp["stars"])-1,5))#store as one hot vector
        
        pickle.dump(review, open(outfile_token, "w+")) #dump review to output file
        
        if len(rating) > 0:
            pickle.dump(np.array(rating), open(outfile_rate, "w+")) #dump rating to output file


def main():
    
    
    # all the input files
    basepath = "/Users/karishma/Dropbox/CMU/fall_2015/MLT/hw5/resources"
    stoplist = "{}/stopword.list".format(basepath)
    
    for name in ["train", "dev", "test", "debug", "dummy"]:
        st = time.time()
        
        inputfile = "{}/yelp_reviews_{}.json".format(basepath, name) #input file path
        reviews = "{}/task1/{}/{}.base_stem".format(basepath, name, name)  # output file path -- reviews
        ratings = "{}/task1/{}/{}.rating_stem".format(basepath, name, name)  # output file path -- ratings
    
        if not os.path.exists(os.path.dirname(reviews)): #construct required dir structure
            os.makedirs(os.path.dirname(reviews))
            
        dr = data_reader(stoplist)
        dr.read_json(inputfile, ratings, reviews, stem=True) #process the data
        
        if name == "train": #tf for dictionary for train data
            mappings = "{}/task1/{}/{}.mapping_stem".format(basepath, name, name)  # output file path -- mappings
            pickle.dump(dr._token_map, open(mappings, "w+"))
        
        print name, "time taken:", time.time() - st
    
if __name__ == "__main__":
    main()
