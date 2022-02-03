# Machine learning program to detect the a submarine is crossing a rock or mine.
# Submarine has sonar which sends sound signal and collects the data to decide whether its travelling on a rock or mine.
# Sonar Data -> Is collected from a lab setup from a metal cylinder and rock.
# Flow:
# Sonar Data -> Feed -> Machine learning model.
# We are going to use Logistic Regression Model as we are using binary mode ROCK or MINE.
# The logistic model is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick.
# This is a supervised model.
# Flow: 
# Sonar Data -> Pre process -> Train and Test split -> Apply Logistic Regression Model -> Becomes Trained Logistic Regression Model <- Test with new sonar data.
# What is logistic regression?
# Logistic regression models the probabilities for classification problems with two possible outcomes. It’s an extension of the linear regression model for classification problems.
# Logistic regression is a statistical method for predicting binary classes. The outcome or target variable is dichotomous in nature. Dichotomous means there are only two possible classes

# Import the depenedencies
import pandas as pd;
import numpy as np;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import accuracy_score;

# Loading data
sonar_data = pd.read_csv('/Sonar_Data.csv',header=None);
# Display rows and cols
# print(sonar_data.shape); 
# Column 16 has the value whether its ROCK(R) or MINE (M)
# print (sonar_data[60].value_counts());
# print (sonar_data.groupby(60).mean());
# print (sonar_data.describe()); # Shows statistical values of the data

# Seperate data X and labels Y
# We are going to make X axis with sonar data 
# Y - axis with 60th columns (R/M)
X = sonar_data.drop(columns=60, axis=1);
Y = sonar_data[60];

# Create Test and Training Data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1);
print (X.shape, X_train.shape, X_test.shape);

# Training the model with Training data
model = LogisticRegression();
model.fit(X_train,Y_train);

# Check the accuracy score of the model. Any model accuracy >75% is good.

X_train_prediction = model.predict(X_train);
training_data_accuracy = accuracy_score(X_train_prediction, Y_train);
print ("Accuracy Score with train data:", (training_data_accuracy)*100);

X_test_prediction = model.predict(X_test);
test_data_accuracy = accuracy_score(X_test_prediction, Y_test);
print ("Accuracy Score with test data:", (test_data_accuracy)*100);

# Now this model can be tested again real sonar data to predict if its ROCK or MINE

input_data = (0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055,0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.499);

# Change the input data to numpy array
input_data_nparray = np.asarray(input_data);

input_data_reshaped = input_data_nparray.reshape(1,-1);

prediction = model.predict(input_data_reshaped);

if prediction[0] == 'M':
  print ("Object is Mine");
else:
  print ("Object is Rock");
