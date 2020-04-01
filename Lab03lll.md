#Lab 03 Assignment
##Yiyan Huang, yh22799
###1.Classification with Dimensionality Reduction

####(a) Dataset preparation and split
I used the wine dataset and digits dataset from sklearn.

```from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

wine = load_wine ()
digits = load_digits()

x_trainW1, x_testW, y_trainW1, y_testW = train_test_split (wine.data, wine.target, test_size = 0.2, random_state=42)
x_trainW, x_valW, y_trainW, y_valW = train_test_split (x_trainW1, y_trainW1, test_size=0.125, random_state=42)
x_trainD1, x_testD, y_trainD1, y_testD = train_test_split (digits.data, digits.target, test_size = 0.2, random_state=42)
x_trainD, x_valD, y_trainD, y_valD = train_test_split (x_trainD1, y_trainD1, test_size=0.125, random_state=42)

```

####(b)train and evaluate two classification algorithms with 10 different features
The wine dataset:

```
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
%matplotlib inline

modelSVM = SVC()
modelKNN = KNeighborsClassifier()
svmAccuracy = []
knnAccuracy = []
component_valueF = []
component_setting = range (1,11)
for component_value in component_setting:
  pca = PCA (n_components = component_value)
  pca.fit(x_trainW)
  x_train_reduced = pca.transform (x_trainW)
  x_val_reduced = pca.transform (x_valW)
  modelSVM.fit(x_train_reduced, y_trainW)
  modelKNN.fit(x_train_reduced, y_trainW)
  svm_accuracy = modelSVM.score(x_val_reduced, y_valW)
  k_accuracy = modelKNN.score(x_val_reduced, y_valW)
  svmAccuracy.append(svm_accuracy)
  knnAccuracy.append(k_accuracy)
  component_valueF.append(component_value)
plt.plot(component_valueF, svmAccuracy, 'g', label = 'SVM accuracy')
plt.plot(component_valueF, knnAccuracy, 'b', label = 'KNN accuracy')
plt.ylabel('Wine accuracy')
plt.xlabel('PCA component value')
plt.legend()
plt.show()
print (svmAccuracy)
print (knnAccuracy)
```
[0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334]

[0.6666666666666666, 0.8888888888888888, 0.8888888888888888, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444, 0.9444444444444444]

The digits dataset:

```
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
%matplotlib inline

modelSVM = SVC()
modelKNN = KNeighborsClassifier()
svmAccuracy = []
knnAccuracy = []
component_valueF = []
component_setting = range (1,11)
for component_value in component_setting:
  pca = PCA (n_components = component_value)
  pca.fit(x_trainD)
  x_train_reduced = pca.transform (x_trainD)
  x_val_reduced = pca.transform (x_valD)
  modelSVM.fit(x_train_reduced, y_trainD)
  modelKNN.fit(x_train_reduced, y_trainD)
  svm_accuracy = modelSVM.score(x_val_reduced, y_valD)
  k_accuracy = modelKNN.score(x_val_reduced, y_valD)
  svmAccuracy.append(svm_accuracy)
  knnAccuracy.append(k_accuracy)
  component_valueF.append(component_value)
plt.plot(component_valueF, svmAccuracy, 'g', label = 'SVM accuracy')
plt.plot(component_valueF, knnAccuracy, 'b', label = 'KNN accuracy')
plt.ylabel('Digits accuracy')
plt.xlabel('PCA component value')
plt.legend()
plt.show()
print (svmAccuracy)
print (knnAccuracy)
```
[0.29444444444444445, 0.6444444444444445, 0.7555555555555555, 0.8722222222222222, 0.9, 0.9333333333333333, 0.9333333333333333, 0.9444444444444444, 0.9777777777777777, 0.9722222222222222]

[0.3055555555555556, 0.6555555555555556, 0.7611111111111111, 0.8333333333333334, 0.9055555555555556, 0.9333333333333333, 0.9555555555555556, 0.9666666666666667, 0.9833333333333333, 0.9833333333333333]

#### (c) Report the predictive performance of each classifier on each dataset as a function of the feature dimension size

Results from code above: 

For the Wine Dataset, the prediction performance (accuracy) of KNN algorithm grows with the increase of number of principal components, whilst the performance of SVM algorithm maintains a similar level across different principal component numbers. When the principal component number reached 10 the KNN algorithm has the best performance in accuracy, reaching 0.94. The accuracy of SVM constantly keeps at 0.83 regardless of the value of feature dimension size.

![](https://i.ibb.co/4FDC89p/1-1.png)

For the Digits Dataset, the prediction performance on accuracy of both KNN and SVM improves with the increase of principal component number. When the principal component number reaches 10, the accuracy of KNN is 0.98. When the princial component number reaches 9, the accuracy of SVM peaks at 0.978, slightly smaller than the best performance of KNN.

![](https://i.ibb.co/ccGsKz2/1-2.png)

#### (d) Write a discussion analyzing the influence of applying PCA on the classification performance. For example, what feature dimension sizes were better/worse and why do you think so? What can you infer by observing the classification performance across the different datasets and different classification algorithms? Your discussion should consist of 2-4 paragraphs
In general, for these two datasets, the prediction performance on accuracy of two algorithms reaches top when the principal component sizes are the largest (except for SVM in the wine dataset, showing a flat line in prediction performance). This means that the more best features the datasets preserve, the best training performance of algorithms. The main reason I think is related with the datasets. The reason lies in that most features in these datasets are useful, and the more features reserved the higher accuracy of predictions by algorithms.

The accuracy of algorithms for the dataset of digist using PCA is higher than that for the dataset of wine. Within each dataset, the accuracy of predictions of KNN is higher than that of SVM. The differences in algorithm accuracy are also results from different datasets. The digits datasets include features extracted from pictures, and therefore PCA is more suitable in this dataset in reducing useless features and improving the accuracy of predictions.

For both datasets KNN has better performance than SVM. The reason might be that the relation between feature data and labels is not a linear one, making KNN an algorithm more suitable for classifying cases.                                                                                                                                                                                              
###2. Classification Using Neural Networks
#### (a) load and split dataset
 
```
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

digits = load_digits()
x_trainVal, x_test, y_trainVal, y_test = train_test_split (digits.data, digits.target, test_size = 0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split (x_trainVal, y_trainVal, test_size=0.125, random_state=42)
```
#### (b) Optimize hyperparameters on the validation set


```
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
stdsc.fit(x_train)
x_train_std = stdsc.transform(x_train)
x_val_std = stdsc.transform(x_val)
x_test_std = stdsc.transform(x_test)

from sklearn.neural_network import MLPClassifier
neurons=[]
for i in range(1,6):
  n_nodes = 100 * i
  accuracy=[]
  hidden_layer=[]
  best_score_n = 0
  for h in range (1,6):
    layer_sizes = [n_nodes]*h
    mlp = MLPClassifier(activation='tanh', h
    idden_layer_sizes = layer_sizes, max_iter=20, verbose=False)
    mlp.fit(x_train_std,y_train)
    accuracy_single=mlp.score(x_val_std,y_val)
    if accuracy_single > best_score_n:
        best_param_n = {'layer number': h}
        best_score_n = accuracy_single
    accuracy.append(accuracy_single)
    hidden_layer.append(h)
  print("best score on training:{:0.5f}".format(best_score_n))
  print('best parameters:{}'.format(best_param_n))   
  neurons.append(n_nodes)
  print('the current number of neurons:', n_nodes)
  plt.plot(hidden_layer, accuracy, label = 'accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('number of layers')
  plt.legend()
  plt.show()
```
best score on training:0.98333

best parameters:{'layer number': 5}

the current number of neurons: 100

![](https://i.ibb.co/WvnmzPM/2-1.png)

best score on training:0.97222

best parameters:{'layer number': 2}

the current number of neurons: 200

![](https://i.ibb.co/hfrpcyF/2-2.png)

best score on training:0.97778

best parameters:{'layer number': 2}

the current number of neurons: 300

![](https://i.ibb.co/V2hwrQd/2-3.png.png)

best score on training:0.97778

best parameters:{'layer number': 1}

the current number of neurons: 400

![](https://i.ibb.co/3yt57G6/2-4.png)

best score on training:0.98333

best parameters:{'layer number': 1}

the current number of neurons: 500

![](https://i.ibb.co/0t5zqDM/2-5.png)

#### (c).Report the optimal hyperparameters you found and the number of weights that are in this optimal model.
The optimal hyperparameters are: 
The number of hidden layers: 1
The number of neurons per layer: 500
Number of weights: the shape of x_train is (1257, 64) with 64 features, and the label classes are 0-9, altogether 10 labels. Therefore the weights are:
64* 500 + 500* 10 = 37000

#### (d) performance of the neural network when using different hyperparameters. For example, what number of hidden layers and neurons per layer did better/worse and why do you think so?
Basically, when the node number is 100, the more layers there are the better performance the algorithm is. When the number of hidden layer is 5 and the node number is 100, the best score on training is 0.98333. When the node number is 200, the accuracy improves at first with the increase of hidden layers and then drops. When the number of hidden layer is 2 and the node number is 200, the best score is achieved reaching 0.97222. When the node number is 300, the overall trend of accuracy resembles the shape of "M", growing at first and peaking when the hidden layer number is 2. The best score is 0.97778. When the node number is 400, the accuracy drops with the increase of hidden layer number and starts to grow from 3 hidden layers. When the node number is 400 and hidden layer number is 1, the best accuracy score is achieved reaching 0.97778. When the node number is 500, the trend of accuracy resembles a shape of "N", but the best performance in accuracy reaches 0.98333 with 1 layer. The best parameters come at 500 in node number and 1 in layer number.

For the five lines by different numbers of nodes, they all apear to get the accuracy rate improved at first but then dropped with the growing of layers. 100 nodes with 5 hidden layers and 500 nodes with 1 layer all delivered best accuracy performance. By contrast, when the node number is quite small and layer number is limited too, the accuracy is quite low, probably because there is underfitting in this stage. Meanwhile, when the node number is already very high, the accuracy is less satisfying for multiple layers, indicating an existence of overfitting. Therefore, when the node number and layer number reach a balance, the MLP classifier has the best performance.


###3.Answerability Classification Using Hand-Crafted Features
#### (a) Access to VizWiz dataset and predefined train/val/test
```
img_dir = "https://ivc.ischool.utexas.edu/VizWiz_visualization_img/"
train_split = 'train'
val_split = 'val'
test_split = 'test'
annotation_file_train = 'https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations/%s.json' %(train_split)
annotation_file_val = 'https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations/%s.json' %(val_split)
annotation_file_test = 'https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations/%s.json' %(test_split)
```
#### (b) Use Azure APIs to extract image-based features and question-based features 

```
#Read the file to extract each dataset example with label
import requests
import numpy as np
from skimage import io
from pprint import pprint
import os
import json

import matplotlib.pyplot as plt
%matplotlib inline

subscription_key = ''
vision_base_url = 'https://centralus.api.cognitive.microsoft.com/vision/v1.0'
vision_analyze_url = vision_base_url + '/analyze?'
```
```
# evaluate an image using Microsoft Vision API
def analyze_image(image_url):
  headers = {'Ocp-Apim-Subscription-key': subscription_key}
  params = {'visualfeatures': 'Adult,Categories,Description,Color, 
  Faces, ImageType, Tags'}
  data = {'url': image_url}
  
  response = requests.post(vision_analyze_url, headers=headers, 
  params=params, json=data)
  
  response.raise_for_status()
  analysis = response.json()
  return analysis

```

```

def extract_image_features(image_url): 
  data_img = analyze_image(image_url)
  
  #number of tags
  tag_number = len(data_img['tags'])

  #confidence score of captions
  if len(data_img['description']['captions']) == 0:
    caption_score = 0
  else:
    caption_score = round(data_img['description']['captions'][0]
    ['confidence'],2)
    
  return [tag_number,caption_score]

```
```

def extract_text_features(question): 
  text_doc = {"documents": [{"id":1, "text": question}]}
  json.dumps(question)

  subscription_key = ''
  endpoint = 'https://centralus.api.cognitive.microsoft.com'
  
  #sentiment value
  sentiment_url = endpoint + "/text/analytics/v2.1/sentiment"
  headers = {"Ocp-Apim-Subscription-Key": subscription_key}
  response = requests.post(sentiment_url, headers=headers, json=text_doc)
  sentiments = response.json()
  sentiment_value = round(sentiments['documents'][0]['score'], 2)
  
  #word numbers of question
  word_number = len(question.split(' '))
  return [sentiment_value, word_number]
``` 
 
 create dataset for training
 
 ```
 split_data = requests.get(annotation_file_train, allow_redirects=True)
n = 0
num_VQs = 850
data_dir = split_data.json()
X = []
y = []
for vq in data_dir[0:num_VQs]:
  # Extracts features decribing the image
  image_name = vq['image']
  image_url = img_dir + image_name 
  image_feature = extract_image_features(image_url)

  # Extracts features decribing the question
  question = vq['question']
  question_feature = extract_text_features(question)

  # Create a multimodal feature to represent both the image and question
  multimodal_features = np.concatenate((question_feature, image_feature))
  
  # Prepare features and labels
  X.append(multimodal_features)
  label = vq['answerable']
  y.append(label)
  n=n+1
  print(n,image_name,question,multimodal_features,label)
 
 ```
 create dataset of validation
 
 ```
 #collect the validation dataset 0-64
split_data = requests.get(annotation_file_val, allow_redirects=True)
n = 0
num_VQs = 121
data_dir = split_data.json()
X_val = []
y_val = []
for vq in data_dir[0:num_VQs]:
  # Extracts features decribing the image
  image_name = vq['image']
  image_url = img_dir + image_name 
  image_feature = extract_image_features(image_url)

  # Extracts features decribing the question
  question = vq['question']
  question_feature = extract_text_features(question)

  # Create a multimodal feature to represent both the image and question
  multimodal_features = np.concatenate((question_feature, image_feature))
  
  # Prepare features and labels
  X_val.append(multimodal_features)
  label = vq['answerable']
  y_val.append(label)
  n=n+1
  print(n,image_name,question,multimodal_features,label)
  
#collect the validation dataset 66-122
split_data = requests.get(annotation_file_val, allow_redirects=True)
n = 66
num_VQs = 122
data_dir = split_data.json()

for vq in data_dir[66:num_VQs]:
  # Extracts features decribing the image
  image_name = vq['image']
  image_url = img_dir + image_name 
  image_feature = extract_image_features(image_url)

  # Extracts features decribing the question
  question = vq['question']
  question_feature = extract_text_features(question)

  # Create a multimodal feature to represent both the image and question
  multimodal_features = np.concatenate((question_feature, image_feature))
  
  # Prepare features and labels
  X_val.append(multimodal_features)
  label = vq['answerable']
  y_val.append(label)
  n=n+1
  print(n,image_name,question,multimodal_features,label)

 ```
collect the test dataset

 ```
split_data = requests.get(annotation_file_test, allow_redirects=True)
n = 0
num_VQs = 100
data_dir = split_data.json()
X_test = []
for vq in data_dir[0:num_VQs]:
  # Extracts features decribing the image
  image_name = vq['image']
  image_url = img_dir + image_name 
  image_feature = extract_image_features(image_url)

  # Extracts features decribing the question
  question = vq['question']
  question_feature = extract_text_features(question)

  # Create a multimodal feature to represent both the image and question
  multimodal_features = np.concatenate((question_feature, image_feature))
  
  # Prepare features and labels
  X_test.append(multimodal_features)
  n=n+1
  print(n,image_name,question,multimodal_features)
 ```
#### (c) Use transformation and classification models to predict whether a visual question is answerable using the input features.
```
#KNN to train and decide hyparameters
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# p=1, this is equivalent to using manhattan_distance (l1)
# p=2, this is euclidean_distance (l2) 
p_setting = range (1,3)
best_score = 0
for p_value in p_setting:
  fold_train_accuracy = []
  best_score_n=0
  neighbor_settings = range (1,11)
  for curKvalue in neighbor_settings:
    clf =  KNeighborsClassifier(n_neighbors = curKvalue, p = p_value)
    clf.fit(X_train, y_train)
    kfold_shuffled = StratifiedKFold(n_splits=5, shuffle=True, 
    random_state=2)
    train_accuracy = cross_val_score(clf, X_train, y_train, 
    cv=kfold_shuffled)
    avg_train_accuracy = train_accuracy.mean()
    fold_train_accuracy.append(avg_train_accuracy)
    if avg_train_accuracy > best_score_n:
      best_param_n = {'n_neighbors': curKvalue}
      best_score_n = avg_train_accuracy
  if best_score_n > best_score:
    best_score = best_score_n
    best_param = {'p': p_value,'n_neighbors': curKvalue}
   
print("best score on training:{:0.2f}".format(best_score))
print('best parameters:{}'.format(best_param))

``` 
best score on training:0.68

best parameters:{'p': 1, 'n_neighbors': 10}

```
#use SVC to train the model with train dataset
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

c_value = 0.1
best_score = 0
svc_accuracy=[]
svmsetting = range (1,6)
for epoch1 in svmsetting:
  gamma_value = 0.001

  for epoch2 in svmsetting:
    degree_value = 1

    for epoch3 in svmsetting:
      svm_clf = SVC(C= c_value,kernel='poly',degree = degree_value, 
                                gamma= gamma_value) 
      svm_clf.fit (X_train,y_train)

      kfold_shuffled = StratifiedKFold(n_splits=5, shuffle=True, 
      random_state=2)
      train_accuracy_before = cross_val_score(svm_clf, X_train, 
      y_train, cv=kfold_shuffled)
      train_accuracy = train_accuracy_before.mean()
      svc_accuracy.append(train_accuracy)
      
      if train_accuracy > best_score:
        best_degree = degree_value
        best_cValue = c_value
        best_gamma = gamma_value
        best_score = train_accuracy

      degree_value = degree_value + 1
    gamma_value = gamma_value + 0.0001
  c_value = c_value + 0.1

print("best accuracy on training:{:0.2f}".format(best_score))
print('best degree:{}'.format(best_degree))
print('best c value:{}'.format(best_cValue))
print('best gamma:{}'.format(best_gamma))
```

best accuracy on training:0.69

best degree:1

best c value:0.1

best gamma:0.001

Gaussian:

```
from sklearn.naive_bayes import GaussianNB

clf_gaussian = GaussianNB ()
clf_gaussian.fit(X_train, y_train)

y_predictedNB = clf_gaussian.predict(X_train)

from sklearn.metrics import classification_report
print ("report for Gaussian:\n", classification_report (y_predictedNB, y_train))
```

report for Gaussian:

               precision    recall  f1-score   support

           0       0.22      0.45      0.30       126
           1       0.88      0.72      0.79       724

    accuracy                           0.68       850

Adabooster:

```
from sklearn.ensemble import AdaBoostClassifier

adabooster = AdaBoostClassifier(n_estimators = 50)
adabooster.fit(X_train, y_train)

kfold_shuffled = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
adabooster_accuracy = cross_val_score(adabooster, X_train, y_train, cv=kfold_shuffled)
avg_adabooster_accuracy = adabooster_accuracy.mean()
print ("accuracy score of adabooster classifier:", avg_adabooster_accuracy)
```
accuracy score of adabooster classifier: 0.6705882352941177

MLP:
 
```
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

neurons=[]

for i in range(1,6):
  n_nodes = 100 * i
  mlp_accuracy=[]
  hidden_layer=[]
  best_score_n = 0
  for h in range (1,6):
    layer_sizes = [n_nodes]*h
    mlp = MLPClassifier(activation='tanh', 
    hidden_layer_sizes = layer_sizes, max_iter=50, 
    verbose=False)
    mlp.fit(X_train,y_train)
    
    kfold_shuffled = StratifiedKFold(n_splits=5, shuffle=True, 
    random_state=2)
    train_accuracy_before = cross_val_score(svm_clf, X_train, 
    y_train, cv=kfold_shuffled)
    train_accuracy = train_accuracy_before.mean()
    mlp_accuracy.append(train_accuracy)
    hidden_layer.append(h)
   
    if train_accuracy > best_score_n:
      best_param_n = {'layer number': h}
      best_score_n = train_accuracy
  print("best score on training:{:0.2f}".format(best_score_n))
  print('best parameters:{}'.format(best_param_n))   
  neurons.append(n_nodes)
  print('the current number of nurons:', n_nodes)
```
best score on training:0.69

best parameters:{'layer number': 1}

the current number of nurons: 500

#### Use the validaiton dataset to determine on the classification system used for the test dataset

Gaussian:

```
from sklearn.naive_bayes import GaussianNB
clf_gaussian = GaussianNB ()
clf_gaussian.fit(X_train, y_train)
y_val_predictedNB = clf_gaussian.predict(X_val)
from sklearn.metrics import classification_report
print ("report for Gaussian:\n", classification_report(y_val_predictedNB, y_val))
```
report for Gaussian:

               precision    recall  f1-score   support

           0       0.21      0.50      0.29        26
           1       0.77      0.47      0.58        94

    accuracy                           0.48       120
    
Adabooster:

```
from sklearn.ensemble import AdaBoostClassifier

adabooster = AdaBoostClassifier(n_estimators = 50)
adabooster.fit(X_train, y_train)
y_adabooster_pred = adabooster.predict(X_val)

from sklearn.metrics import classification_report
print ("report for Adabooster:\n", classification_report(y_adabooster_pred, y_val))
```
report for Adabooster:

               precision    recall  f1-score   support

           0       0.19      0.71      0.30        17
           1       0.91      0.50      0.65       103

    accuracy                           0.53       120
    
SVC:

```
from sklearn.svm import SVC
svm_clf = SVC(C= 0.1,kernel='poly',degree = 1, gamma= 0.001) 
svm_clf.fit (X_train,y_train)
y_val_svm=svm_clf.predict(X_val)
from sklearn.metrics import classification_report
print ("report for SVC:\n", classification_report(y_val_svm, y_val))
```
report for SVC:

               precision    recall  f1-score   support

           0       0.00      0.00      0.00         0
           1       1.00      0.47      0.64       120

    accuracy                           0.48       120

KNN:

```
from sklearn.neighbors import KNeighborsClassifier
clf_KNN =  KNeighborsClassifier(n_neighbors = 10, p = 1)
clf_KNN.fit(X_train, y_train)
y_val_KNN=clf_KNN.predict(X_val)
from sklearn.metrics import classification_report
print ("report for KNN:\n", classification_report(y_val_KNN, y_val))
```
report for KNN:

               precision    recall  f1-score   support

           0       0.32      0.69      0.43        29
           1       0.84      0.53      0.65        91

    accuracy                           0.57       120

MLP:

```
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation='tanh', hidden_layer_sizes = (500,), max_iter=50, verbose=False)
mlp.fit(X_train, y_train)
y_val_MLP = mlp.predict(X_val)
from sklearn.metrics import classification_report
print ("report for MLP:\n", classification_report(y_val_MLP, y_val))
```
report for MLP:

               precision    recall  f1-score   support

           0       0.24      0.68      0.35        22
           1       0.88      0.51      0.65        98

    accuracy                           0.54       120
   
#### (d) Use the classification systems to make predictions on the test dataset

```
import csv
from sklearn.neighbors import KNeighborsClassifier
clf_KNN =  KNeighborsClassifier(n_neighbors = 10, p = 1)
clf_KNN.fit(X_train, y_train)
y_predict_test=clf_KNN.predict(X_test)

# f = open("results.csv", mode="w")
with open("/content/drive/My Drive/Colab Notebooks/PredictionResults.csv", mode="w") as f:
  results = csv.writer(f)
  for prediction in y_predict_test:
    results.writerow([prediction])
```

#### (e) Write 2-4 paragraphs describing your proposed prediction method. Describe the implementation of your proposed approach in such a way that a 1) reader could reproduce your set-up and 2) understand why you made your design decisions.

The purpose is to estimate whether a question could be answered based on the proposed questions and the images taken. Below are the steps I take to realize this way:

1) Build up the dataset with features pertaining to prediction results

I use Microsoft Azure to extract features of the images and questions asked. Based on the user guidebook of Microsoft, it would analyze the color, adult content, face, tags based on object recognition and description etc. of images. Specifically, the tags include objects identified by Azure and the description contains an automatically generated captions. Based on a rough review of the labeled dataset for training, for those visual questions that could not be answered there might be serious image problems, such as the images are too blurry to identify objects or the target objects were not shooted at all. Therefore I chose a) the number of entries under tags as one feature, under such a premise that the blurrier the image is the less objects could be identified (less tags), and b) the confidence score of captions generated as the other feature, under such a premise that the more vague the image is the less confident Azure is in generating the caption.

For the text features, Azure could analyze the sentiment, entities, language detection and extract key phrases. In my opinion, the reason why a question could not be answered might be that the users failed to propose a question, but proposed a sentence instead. Or the question proposed is not specific enough to be answered. The features of Azure text analytics are not connected that closely, but the sentiment score could provide a reference whether users are making a judgement or jokes or proposing serious questions. To measure whether the question is specific enough, I used the number of words within the sentence as the other feature.

2) Divide up the dataset 

I withdrew 850 entries from VizWiz dataset for training, and used cross validation to identify the optimal hyperparameters. In line with a 70%/10% ratio for training and validation dataset, I withdreww 120 entries from the validation split of VizWiz as the validation dataset. Following analysis above, I withdrew the features of number of tags, caption confidence score, sentiment score and number of words within sentences are the features of the transformed datasets for training and validation.
#### (f) Write 2-4 paragraphs describing the analysis you conducted with the training and/or validation datasets that guided your choice for your prediction system design (e.g., hyperparameters, classification models, etc).

As the data include cardinal numbers and nominal numbers based on my premises, and the labels are nominal numbers, I mainly considered algorithms for classification, including Gaussian, SVM, KNN, MLP, and Adabooster to complete the prediction. 

For the **hyperparmeters**, I used cross validation in the training dataset to determine the optimal hyperparameters for KNN, MLP, and SVM with attention paid to the best accuracy gained in training:

As there are only four features and the data is merely cardinal or nominal numbers, they can be directly put into use. I tried standardization to transform the data once but found the accuracy is lower than before. Therefore, I didn't use any transformation algorithm to process the dataset.

The results of cross validation for different algorithms are as follows. 


               Algorithm           BestParameters                BestAccuracy

           1     KNN       {'p': 1, 'n_neighbors': 10}                 0.68
           2     SVC       {'degree':1, 'c_value': 0.1, 'gamma':0.001} 0.69
           3     MLP       {'layer number': 1, 'node number': 500}     0.69
           4   Adabooster             default                          0.67
           5   Gaussian               default                          0.68



For the **selection of prediction system**, I used the validation dataset to measure the accuracy of predictions for each trained algorithm. The results are like follows:


               Algorithm                 BestAccuracy

           1     KNN                         0.57
           2     SVC                         0.48
           3     MLP                         0.54
           4   Adabooster                    0.53
           5   Gaussian                      0.48

Based on the training results, the accuracy of all 5 algorithms are quite near, realizing an accuracy around 0.68 even though I tried different hyperparameters. This indicates that the best etimation performance on this dataset is around 0.7 within the algorithms I used. Then I used the trained algorithm to testify the validation dataset with their best hyperparameters, and found out KNN has the best performance in front of a new dataset, reaching 0.57, followed by MLP, Adabooster, SVC and Gaussian. The reasons for loss in accuracy might be: 1) features selected are not quite related with the labels as the accuracy in training is not quite high, 2) inadequate data sample, probably the larger dataset for training is the higher performance of accuracy, 3) these algorithm might not be the best algorithms to make predictions, and 4) there are less useful features in the dataset and should be screened out. 

Still, through comparison, KNN shows the best accuracy with hyperparameters of p=1, n_neighbors=10. So I decided to use KNN to make predictions on the test dataset.