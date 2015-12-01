import numpy as np


def norm(data):
    """
        Parameters
        ---------
        data: ndarray
            input array which is to be normalized
        Returns
        -------
        ndarray:
            normalized array
    """
    sum_data = np.sum(data)  
    return np.divide(data, sum_data)

def onehot(val,size):
    """
        Parameters
        ----------
        val: int
            value to be converted to one hot vector
        size: int
            size of the one hot vector
        
        Returns
        -------
        arr: ndarray
            one hot vector
    """
    arr = np.zeros(size)
    arr[val]=1
    return arr
    
