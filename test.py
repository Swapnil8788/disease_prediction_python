# Importing the libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the dataset
training_dataset = pd.read_csv('Training.csv')
test_dataset = pd.read_csv('Testing.csv')

# Slicing and Dicing the dataset to separate features from predictions
X = training_dataset.iloc[:, 0:132].values
Y = training_dataset.iloc[:, -1].values

# Dimensionality Reduction for removing redundancies
dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()

# Encoding String values to integer constants
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(Y)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Implementing the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Saving the information of columns
cols     = training_dataset.columns
cols     = cols[:-1]

# Checking the Important features
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Implementing the Visual Tree
from sklearn.tree import _tree

# Method to simulate the working of a Chatbot by extracting and formulating questions
def print_disease(node):
        #print(node)
        node = node[0]
        #print(len(node))
        val  = node.nonzero() 
        #print(val)
        disease = labelencoder.inverse_transform(val[0])
        return disease
def recurse(node, depth):
            global val,ans
            global tree_,feature_name,symptoms_present
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                yield name + " ?"
                
#                ans = input()
                ans = ans.lower()
                if ans == 'yes':
                    val = 1
                else:
                    val = 0
                if  val <= threshold:
                    yield from recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    yield from recurse(tree_.children_right[node], depth + 1)
            else:
                strData=""
                present_disease = print_disease(tree_.value[node])
#                print( "You may have " +  present_disease )
#                print()
                strData="You may have :" +  str(present_disease)
               
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
                
                red_cols = dimensionality_reduction.columns 
                symptoms_given = red_cols[dimensionality_reduction.loc[present_disease].values[0].nonzero()]
#                print("symptoms present  " + str(list(symptoms_present)))
#                print()
                strData="symptoms present:  " + str(list(symptoms_present))
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
#                print("symptoms given "  +  str(list(symptoms_given)) )  
#                print()
                strData="symptoms given: "  +  str(list(symptoms_given))
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
                confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
#                print("confidence level is " + str(confidence_level))
#                print()
                strData="confidence level is: " + str(confidence_level)
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
#                print('The model suggests:')
#                print()
                strData='The model suggests:'
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
                row = doctors[doctors['disease'] == present_disease[0]]
#                print('Consult ', str(row['name'].values))
#                print()
                strData='Consult '+ str(row['name'].values)
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
#                print('Visit ', str(row['link'].values))
                #print(present_disease[0])
                # hyperlink = HyperlinkManager(QuestionDigonosis.objRef.txtDigonosis)
                strData='Visit '+ str(row['link'].values[0])
                def click1():
                    webbrowser.open_new(str(row['link'].values[0]))
                # QuestionDigonosis.objRef.txtDigonosis.insert(INSERT, strData, hyperlink.add(click1))
                #QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
                yield strData
        
def tree_to_code(tree, feature_names):
        global tree_,feature_name,symptoms_present
        tree_ = tree.tree_
        #print(tree_)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        #print("def tree({}):".format(", ".join(feature_names)))
        symptoms_present = []   
#        recurse(0, 1)
    

def execute_bot():
#    print("Please reply with yes/Yes or no/No for the following symptoms")    
    tree_to_code(classifier,cols)
    print("hello world")



# This section of code to be run after scraping the data

doc_dataset = pd.read_csv('doctors_dataset.csv', names = ['Name', 'Description'])


diseases = dimensionality_reduction.index
diseases = pd.DataFrame(diseases)

doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan

doctors['disease'] = diseases['prognosis']


doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']

record = doctors[doctors['disease'] == 'AIDS']
record['name']
record['link']




# Execute the bot and see it in Action
# execute_bot()
























# This section of code to be run after scraping the data

doc_dataset = pd.read_csv('doctors_dataset.csv', names = ['Name', 'Description'])


diseases = dimensionality_reduction.index
diseases = pd.DataFrame(diseases)

doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan

doctors['disease'] = diseases['prognosis']


doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']

record = doctors[doctors['disease'] == 'AIDS']
record['name']
record['link']




# Execute the bot and see it in Action
execute_bot()