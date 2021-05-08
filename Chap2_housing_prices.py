#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import requirments
import os
import tarfile
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
#import custom transformer class
from custom_transformer import CombinedAttributesAdder


# In[ ]:


#define contants
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# In[ ]:


#Get housing data from internet, make dirs, extract datasets
def fetch_housing_data(housing_url = HOUSING_URL,housing_path = HOUSING_PATH):
    os.makedirs(housing_path,exist_ok = True)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
    

#load dataset into variable
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#Niave method to split the dataset into testing and training
def split_train_test(data, ratio):
    #shuffled list of indices for each object in dataset
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    #Calculate how many objects should be in the test set
    test_size = int(len(data)*ratio)
    #grab every element with index less than test size
    test_indices = shuffled_indices[:test_size]
    #grab every element with index greater than test size
    train_indices = shuffled_indices[test_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[ ]:


#gather and load california housing dataset
fetch_housing_data(housing_url = HOUSING_URL,housing_path = HOUSING_PATH)
housing = load_housing_data(housing_path = HOUSING_PATH)


# # DATA EXPLORATION

# In[ ]:


#print summary of housing dataset
housing.head()


# In[ ]:


#print attribute information
housing.info()


# In[ ]:


#count missing values in each column
print(housing.isna().sum())
print(f"Only {(207/20640): .2%} of toatal_bedrooms is missing")


# In[ ]:


#Get descriptive stats
housing.describe()


# In[ ]:


#Hist the numerical attrs
housing.hist(bins = 50, figsize = (20,15));


# In[ ]:


#Split into testing and training using stratiefied sampling on the income attribute

#create a discrete version of median_income for the strata
housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0.0,1.5,3.0,4.5,6,np.inf],
                              labels = [1,2,3,4,5])
#plot new attr
housing["income_cat"].hist();


# In[ ]:


#plot geodpatial data
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4,
            s = housing["population"]/100, label = "population", figsize = (10,7),
            c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True)
plt.legend();


# In[ ]:


#Explore how the attributes are correlated to the target attribute
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# In[ ]:


#plot house prices vs median income
housing.plot(kind = "scatter", x= "median_income", y = "median_house_value", alpha  = 0.1);


# In[ ]:


#Explore diff attr combinations 
housing["beds_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
#Explore how the attributes are correlated to the target attribute
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# # PREPROCESSING

# In[ ]:


#Split The Data Into Test And Train


#create split object
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

#take a representative sample of the pop for both test and train.
for train_index, test_index in split.split(housing,housing["income_cat"]):
    train_set = housing.iloc[train_index]
    test_set = housing.iloc[test_index]
    
#remove the income catagory attribute from test and training sets
for set_ in (train_set, test_set):
    set_.drop("income_cat", axis = 1, inplace = True)


# In[ ]:


#first split the target off from the other attributes in training set
housing_training = train_set.drop("median_house_value", axis = 1)
housing_labels = train_set["median_house_value"].copy()


#first split the target off from the other attributes in test set
housing_test = test_set.drop("median_house_value", axis = 1)
housing_labels_test = test_set["median_house_value"].copy()


# In[ ]:


#total_bedrooms has missing data. We will impute the missing data.
#initialize imputer object
imputer = SimpleImputer(strategy = "median")
#drop catagorical attrs from train/test
housing_num = housing_training.drop("ocean_proximity", axis = 1)
housing_num_test = housing_test.drop("ocean_proximity", axis = 1)
#fit the imputer on the train set and transorm train/test
imputer.fit(housing_num)
X_train = imputer.transform(housing_num)
X_test = imputer.transform(housing_num_test)
#convert training set back to dataframe
housing_tr = pd.DataFrame(X_train,columns = housing_num.columns,
                         index = housing_num.index)

#convert test set back to dataframe
housing_tr_test = pd.DataFrame(X_test,columns = housing_num_test.columns,
                         index = housing_num_test.index)


# In[ ]:


#train/test now contain all of the numerical attrs with no missing data
#Housing_tr excludes Ocean Proximity, which is a string catagorical value
#Best to convert this to a number for the algorithms


#get catagorical attributes
housing_cat = housing_training[["ocean_proximity"]]
housing_cat_test = test_set[["ocean_proximity"]]


#apply one hot encoding to the catagorical attr and append it to the rest of the training attrs
housing_cat_1hot = pd.get_dummies(housing_cat, prefix=["ocean_proximity"], columns = ["ocean_proximity"], drop_first=True)
housing_cat_1hot_test = pd.get_dummies(housing_cat_test, prefix=["ocean_proximity"], columns = ["ocean_proximity"], drop_first=True)

#add the encoded attrs back to the training set
attrs1 = [housing_tr, housing_cat_1hot]
train = pd.concat(attrs1, axis = 1)

#add the encoded cat attrs and target back to the testing set
attrs2 = [housing_tr_test, housing_cat_1hot_test,housing_labels_test]
test = pd.concat(attrs2, axis = 1)


# In[ ]:


#training and test set now have no missing data 
#and are fully numerical
print(train.info())
print(test.info())


# In[ ]:


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(train.values)


# In[ ]:




