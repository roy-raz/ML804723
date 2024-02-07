import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = pd.value_counts(data[:,-1])
    probs = labels/data.shape[0]
    gini += 1-np.sum(np.square(probs))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = pd.value_counts(data[:,-1])
    #print(np.max(labels))
    probs = labels/data.shape[0]
    #print(probs)
    entropy += -np.sum(probs * np.log2(probs))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ##########################################################################
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    df=pd.DataFrame(data)
    total_length=df.shape[0]
    group_by=df.groupby(feature)
    goodness += impurity_func(df.to_numpy())
    split=1
    i=0
    if(gain_ratio):
        impurity_func=calc_entropy
        labels = pd.value_counts(data[:,feature])
        probs = labels/total_length
        split = -np.sum(probs * np.log2(probs))
    for attribute, sub_df in group_by:
        sub_df_length=sub_df.shape[0]
        groups.update({attribute:sub_df})
        goodness -= impurity_func(sub_df.to_numpy())*sub_df_length/total_length
    if(goodness==0.0 or split==0):
        split=1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness/split, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio


    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels=pd.value_counts(self.data[:,self.feature])
        pred=labels.idxmax()
        #print(labels[0,:])
        #print(pred)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        #selecting best feature to split by
        best_goodness=0
        best_split=[]
        for feature in range(self.data.shape[1]-1):
            goodness,split=goodness_of_split(self.data,feature,impurity_func,self.gain_ratio)
            #print(goodness)
            if(best_goodness<goodness):
                self.feature=feature
                best_goodness=goodness
                best_split=split
        #Split data now according to best feature
        #print(best_split)
        #Special case in which all rows are the same the classes are different
        if best_goodness==0:
            self.terminal=True
            return
        itemized_split=best_split.items()
        #CHI Square
        if len(itemized_split) >= 2:
            old_labels=pd.value_counts(self.data[:,-1])
            chi_val=0
            for key,value in itemized_split:
                chi_arr_value=value.to_numpy()
                new_labels=pd.value_counts(chi_arr_value[:,-1])
                p_n_ratio=new_labels/chi_arr_value.shape[0]
                #print(f"p_n ratio is {p_n_ratio}")
                e_ratio=chi_arr_value.shape[0]*old_labels/self.data.shape[0]
                #print(f"e ratio is {e_ratio}")
                sub_ratio=p_n_ratio-e_ratio
                sub_ratio=sub_ratio.fillna(value=e_ratio*-1)
                chi_val+=np.sum(np.square(sub_ratio)/e_ratio)
                #print(chi_val)
            #Check now the chi value
            df=chi_table[len(itemized_split)]
            #print(df)
            #print(type(df))
            if self.chi in df.keys():
                val_risk=df[self.chi]
                if chi_val<val_risk:
                    self.terminal=True
                    return




        #print(itemized_split)
        #print(type(best_split))
        for key,value in itemized_split:
            arr_value=value.to_numpy()
            child=DecisionNode(data=arr_value,depth=self.depth+1,max_depth=self.max_depth,chi=self.chi,gain_ratio=self.gain_ratio)
            self.add_child(child,key)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    root=DecisionNode(data=data,gain_ratio=gain_ratio, chi=chi, max_depth=max_depth)
    #print (f'building a tree of depth {max_depth} and chi of {chi}')
    build_subtree(root,impurity, gain_ratio, chi, max_depth)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

def build_subtree(node,impurity, gain_ratio, chi, max_depth):
    if impurity(node.data)==0.0 or node.depth>=node.max_depth:
        node.terminal=True
        return
    else:
        node.split(impurity_func=impurity)
        for child in node.children:
            build_subtree(child,impurity, gain_ratio, chi, max_depth)
def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    current_node = root
    while current_node.terminal == False:
        test_feature = current_node.feature
        try:
            current_node = current_node.children[current_node.children_values.index(instance[test_feature])]
        except ValueError:
            return current_node.pred
    pred = current_node.pred

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    correct = 0
    for i in range(dataset.shape[0]):
        if predict(node,dataset[i,:]) == dataset[i,-1]:
            correct += 1
    accuracy = float(correct)/dataset.shape[0]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#    for depth in [1, 2, 3, 4]:
        print (depth)
        depth_tree=build_tree(data=X_train,impurity=calc_entropy,gain_ratio=True,max_depth=depth)
        training.append(calc_accuracy(depth_tree,X_train))
        testing.append(calc_accuracy(depth_tree,X_test))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    for chi_value in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        chi_tree=build_tree(data=X_train,impurity=calc_entropy,gain_ratio=True,chi=chi_value)
        chi_training_acc.append(calc_accuracy(chi_tree,X_train))
        chi_testing_acc.append(calc_accuracy(chi_tree,X_test))
        depth.append(calc_depth(chi_tree))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth

def calc_depth(root):

    if root.terminal:
        return root.depth
    else:
        return np.max([calc_depth(children) for children in root.children])
    

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if node.terminal:
        n_nodes = 1
    else:
        n_nodes = np.sum([count_nodes(children) for children in node.children]) + 1
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes
def print_tree(node, depth=0, parent_feature='ROOT', feature_val='ROOT'):
    '''
    prints the tree according to the example above

    Input:
    - node: a node in the decision tree

    This function has no return value
    '''
    if node.terminal == False:
        if node.depth == 0:
            print('[ROOT, feature=X{}]'.format(node.feature))
        else:
            print('{}[X{}={}, feature=X{}], Depth: {}'.format(depth*'  ', parent_feature, feature_val,
                                                              node.feature, node.depth))
        for i, child in enumerate(node.children):
            print_tree(child, depth+1, node.feature, node.children_values[i])
    else:
        classes_count = {}
        labels, counts = np.unique(node.data[:, -1], return_counts=True)
        for l, c in zip(labels, counts):
            classes_count[l] = c
        print('{}[X{}={}, leaf]: [{}], Depth: {}'.format(depth*'  ', parent_feature, feature_val,
                                                         classes_count, node.depth))







#%%