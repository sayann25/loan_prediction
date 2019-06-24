
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest,chi2,f_classif
get_ipython().run_line_magic('matplotlib', 'inline')



def remove_outlier(df_in,col_name):
    q1=df_in[col_name].quantile(0.25)
    q3=df_in[col_name].quantile(0.75)
    iqr=q3-q1
    f_low=q1-1.5*iqr
    f_high=q3+1.5*iqr
    df_out=df_in.loc[(df_in[col_name]>f_low) & (df_in[col_name]<f_high)]
    return df_out

def fresh_outlier(df_in,col_name):
        q1=df_in[col_name].quantile(0.25)
        q3=df_in[col_name].quantile(0.75)
        iqr=q3-q1
        f_low=q1-1.5*iqr
        f_high=q3+1.5*iqr
        df_out=df_in[col_name].loc[(df_in[col_name]<f_low) | (df_in[col_name]>f_high)]
        #print(df_out)
        for i in df_out:
            df_in[col_name]=df_in[col_name].replace( i,np.NaN)
            mean=int(df_in[col_name].mean(skipna=True))
            df_in[col_name]=df_in[col_name].replace(np.NaN,mean)   
        return df_in



df = pd.read_csv('DB.csv')
print (df.shape)
df.head(5)



df.isnull().count()



df.describe()



df.info()



sns.heatmap(df.corr(),annot=True,cmap='BuPu',annot_kws={"size": 10},linewidths=.9,linecolor='black')

df.corr()


sns.countplot(x="Duration_in_Months",data=df)

sns.countplot(x="Credit_Amount",data=df)

sns.countplot(x="Age",data=df)



plt.hist(df['Age'], histtype = "bar", rwidth = 0.8,)
plt.xlabel ("Age groups")
plt.ylabel ("Number of people")
plt.title("Histrogram")
plt.show()

plt.hist(df['Credit_Amount'], histtype = "bar", rwidth = 0.8,)
plt.xlabel ("Credit Amount")
plt.ylabel ("Number of people")
plt.title("Histrogram")
plt.show()

plt.hist(df['Duration_in_Months'], histtype = "bar", rwidth = 0.8,)
plt.xlabel ("Duration")
plt.ylabel ("Number of people")
plt.title("Histrogram")
plt.show()



sns.distplot(df["Duration_in_Months"])

sns.distplot(df["Credit_Amount"])

sns.distplot(df["Age"])



sns.boxplot(df["Duration_in_Months"])

sns.boxplot(df["Credit_Amount"])

sns.boxplot(df["Age"])



from sklearn.preprocessing import LabelEncoder
le_Dummy = LabelEncoder()
df['Status_Checking_Acc_N'] = le_Dummy.fit_transform (df['Status_Checking_Acc'])
df['Credit_History_N'] = le_Dummy.fit_transform (df['Credit_History'])
df['Purposre_Credit_Taken_N'] = le_Dummy.fit_transform (df['Purposre_Credit_Taken'])
df['Savings_Acc_N'] = le_Dummy.fit_transform (df['Savings_Acc'])
df['Years_At_Present_Employment_N'] = le_Dummy.fit_transform (df['Years_At_Present_Employment'])
df['Marital_Status_Gender_N'] = le_Dummy.fit_transform (df['Marital_Status_Gender'])
df['Other_Debtors_Guarantors_N'] = le_Dummy.fit_transform (df['Other_Debtors_Guarantors'])
df['Property_N'] = le_Dummy.fit_transform (df['Property'])
df['Other_Inst_Plans_N'] = le_Dummy.fit_transform (df['Other_Inst_Plans '])
df['Housingn_N'] = le_Dummy.fit_transform (df['Housing'])
df['Job_N'] = le_Dummy.fit_transform (df['Job'])
df['Telephone_N'] = le_Dummy.fit_transform (df['Telephone'])
df['Foreign_Worker_N'] = le_Dummy.fit_transform (df['Foreign_Worker'])

print (df.head())



df1 = df.drop(['Status_Checking_Acc', 'Credit_History', 'Purposre_Credit_Taken', 'Savings_Acc', 'Years_At_Present_Employment','Marital_Status_Gender', 'Other_Debtors_Guarantors', 'Property', 'Housing', 'Job', 'Telephone','Foreign_Worker', 'Customer_ID', 'Other_Inst_Plans ',"Count"], axis =1)

print (df1.head())



sns.heatmap(df1.corr(),cmap='BuPu',annot=True,linewidths=.9,linecolor='black')


df1.corr()



df2=fresh_outlier(df1,'Age')
df3=fresh_outlier(df2,'Credit_Amount')
df4=fresh_outlier(df3,'Duration_in_Months')


sns.boxplot(df4["Credit_Amount"])

sns.boxplot(df4["Duration_in_Months"])

sns.boxplot(df4["Age"])



x=df4.drop('Default_On_Payment',axis=1)
y=df4['Default_On_Payment']



df5=df4[['Age',"Duration_in_Months",'Credit_Amount']]
df6=x.drop(['Age',"Duration_in_Months",'Credit_Amount'],axis=1)
df7=df6.reset_index()
df7.head()



sc_x=StandardScaler()

Xdata=sc_x.fit_transform(df5)

Xdata2=pd.DataFrame(Xdata)
df8 = pd.concat([df7,Xdata2],axis=1)
df8=df8.drop("index",axis=1)
df8.head()


bestfeatures=SelectKBest(score_func=f_classif,k=2)

fit=bestfeatures.fit(Xdata,y)
dfscores=pd.DataFrame(fit.scores_)
print(dfscores)
X=pd.DataFrame(df8)
X.head()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


model=["LR",'DT','KNN', 'NB']

score,model_name = 0,""

for i in model:
    if i=="LR":
        logmodel = LogisticRegression()
        logmodel.fit(X_train, y_train)
        predictions = logmodel.predict(X_test)
        res = confusion_matrix(y_test, predictions)
        print('\n The Confusion Matrix of Logistic Regression is : \n', res)
        
        if score<f1_score(y_test, predictions):
            score = f1_score(y_test, predictions)
            model_name="Logistic Regression."
        print('\n Accuracy of Logistic Regression Model is : ',round(accuracy_score(y_test, predictions)*100,2),'%')
        print("\n Logistic Regression Model Report : \n")
        print(classification_report(y_test, predictions))
        
        
    elif i=="DT":
        classifier_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth = 3, min_samples_leaf=5)
        classifier_entropy.fit(X_train, y_train)
        y_pred = classifier_entropy.predict(X_test)
        res = confusion_matrix(y_test, y_pred)
        print('\n The Confusion Matrix of Decision Tree is : \n', res)
        
        if score<f1_score(y_test, y_pred):
            score = f1_score(y_test, y_pred)
            model_name="Decision Tree."
        print('\n Accuracy of Decision Tree is : ', round(accuracy_score(y_test, y_pred)*100,2),'%')
        print("\n Decision Tree Report : \n")
        print(classification_report(y_test, y_pred))

        
    elif i=="KNN":
        classifier= KNeighborsClassifier(n_neighbors=31,p=2,metric='euclidean')
        classifier.fit(X_train,y_train)
        y_pred2 =classifier.predict(X_test)
        res = confusion_matrix(y_test, y_pred2)
        print('\n The Confusion Matrix of KNN Model is : \n',res)
        
        if score<f1_score(y_test,y_pred2):
            score = f1_score(y_test,y_pred2)
            model_name="K-Nearest Neighbor Model."
        print('\n Accuracy of KNN Model is : ', round(accuracy_score(y_test, y_pred2)*100,2),'%')
        print("\n K-Nearest Neighbor Model Report : \n")
        print(classification_report(y_test,y_pred2))
        
        
    elif i == 'NB':
        model = GaussianNB()
        model.fit(X_train,y_train)
        predicted = model.predict(X_test)
        res = confusion_matrix(y_test, predicted)
        print('\n The Confusion Matrix of Naive Bayes Model is : \n', res)
        
        if score<f1_score(y_test, predicted):
            score = f1_score(y_test, predicted)
            model_name="Naive Bayes Model."
        print('\n Accuracy of Naive Bayes Model is :', round(accuracy_score(y_test, predicted)*100,2),'%')
        print("\n Naive Bayes Model Report : \n")
        print(classification_report(y_test, predicted))
    print('----------------------------------------------------------------------------------------------------')

print("\n The Highest f1_score is : ",score,'\n The Best Fitted Model based on f1_score is : ',model_name)          