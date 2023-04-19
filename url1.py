import pickle

import pandas as pd
import numpy as np
import random
# from google.colab import files
import io 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


# upload = files.upload()

urls_data = pd.read_csv("urldata.csv")


type(urls_data)

# print(urls_data.tail())

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens

y = urls_data["label"]

# Features
url_list = urls_data["url"]

vectorizer = TfidfVectorizer(tokenizer=makeTokens)

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
#using logistic regression
logit = LogisticRegression()	
logit.fit(X_train, y_train)

# print (url_list.tail())


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

print("Accuracy ",logit.score(X_test, y_test))

X_predict = ["google.com/search=jcharistech",
"google.com/search=faizanahmad",
"pakistanifacebookforever.com/getpassword.php/", 
"www.radsport-voggel.de/wp-admin/includes/log.exe", 
"ahrenhei.without-transfer.ru/nethost.exe ",
"www.itidea.it/centroesteticosothys/img/_notes/gum.exe"]

X_predict = ["https://github.com/"]

X_predict = vectorizer.transform(X_predict)
New_predict = logit.predict(X_predict)

print(New_predict)

# X_predict1 = ["www.buyfakebillsonlinee.blogspot.com", 
# "www.unitedairlineslogistics.com",
# "www.stonehousedelivery.com",
# "www.silkroadmeds-onlinepharmacy.com" ]

# X_predict1 = vectorizer.transform(X_predict1)
# New_predict1 = logit.predict(X_predict1)
# print(New_predict1)

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(url_list)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# logit = LogisticRegression()	#using logistic regression
# logit.fit(X_train, y_train)

# print("Accuracy ",logit.score(X_test, y_test))



manas_model = logit 


with open('manas_model.pkl', 'wb') as f:
    pickle.dump(manas_model,f)

with open('manas_model.pkl', 'rb') as f:
    model =pickle.load(f)


# Define a new example input
new_data =np.array(['https://github.com/'])
new_data = vectorizer.transform(new_data)

# Use the loaded model to make predictions on the new input
predicted_labels = manas_model.predict(new_data)
# x_2d = predicted_labels.reshape(-1, 1)

# y_pred = model.predict(x_2d)
# import numpy as np

# # Define a 1D array
# x = np.array(['https://github.com/'])

# # Reshape the array to a 2D array with a single feature
# x_2d = x.reshape(-1, 1)

# # Make a prediction using the 2D array
# y_pred = model.predict(x_2d)

print(predicted_labels)



# print(y_pred)

