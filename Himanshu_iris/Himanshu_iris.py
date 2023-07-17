import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ----> read the data file and covert it to dataframe and then use it as dataframe
df = pd.read_csv("iris.data",names=["sepal length in cm","sepal width in cm","petal length in cm","petal width in cm","class"])

# ----> checking the shape of the dataframe
print(df.shape)

# ----> checking information about variables or columns
print(df.info())

# ----> checking descriptive analysis of numerical data 
print(df.describe())

# ----> checking descriptive analysis of categorical data
print(df["class"].describe())

# ----> checking null values count and visualize them 
print(df.isnull().sum())
plt.subplots()
plt.subplot(1,2,1)
df.isnull().sum().plot(kind="bar")
plt.subplot(1,2,2)
sns.heatmap(data=df.isnull(),cmap="viridis")
plt.show()

# ----> conclusion that no null value in dataset 

# ----> handling target variable i.e class

# ----> checking count of unique values
print(df["class"].value_counts())

# ----> as values are uniformly distributed so no further checking needed 

# ----> checking outliers
plt.subplots() 
for i,a in enumerate(df.drop(columns="class").columns):
    plt.subplot(2,2,i+1)
    sns.boxplot(data=df[a],color="lightgreen")
    plt.title(a)
plt.show()

# ----> treating outliers 
def outlier_treatment(data,how):
    low = data.quantile(.25)-1.5*(data.quantile(.75)-data.quantile(.25))
    up = data.quantile(.75)+1.5*(data.quantile(.75)-data.quantile(.25))
    if how == "mode":
        data[data>up],data[data<low]=data.mode(),data.mode()
    elif how == "mean":
        data[data>up],data[data<low]=data.mean(),data.mean()
    elif how == "lim":
        data[data>up],data[data<low]=up,low
        
    elif how == "median":
        data[data>up],data[data<low]=data.median(),data.median()
    else:
        None
    return data

# ----> after checking box plot outlier was detected in sepal width in cm columns so treatment for that 
df["sepal width in cm"] = outlier_treatment(data=df["sepal width in cm"],how="lim")

# ----> after treatment visuaize outlier
sns.boxplot(df["sepal width in cm"],color="lightgreen")
plt.title("sepal width in cm after treatment")
plt.show()

# ----> checking relation between target and independent variables 
x = df.drop(columns="class")
y = df["class"]
sns.pairplot(data=df,hue="class")
plt.show()

################## ----> uncomment below code to remove multicoleanirity from the dataset    #############################
# ----> # ----> checking correlations between independent variables
# ----> sns.heatmap(x.corr(),annot=True,cmap="viridis")
# ----> plt.show()
# ----> cor = np.unique([sorted([a,b]) for a in x.columns for b in x.columns if a!=b and abs(df[a].corr(df[b]))>0.5],axis=0)
# ----> print(cor)

# ----> # ----> function for removing multicolinearity using variance inflation factor method 
# ----> def vif_remover(data):
# ---->     vif = pd.Series([variance_inflation_factor(data.values,i) for i in range(data.shape[1])],index=data.columns)
# ---->     if vif.max()>5:
# ---->         data.drop(columns=vif[vif==vif.max()].index[0],inplace=True)
# ---->         print(f"dropped {vif[vif==vif.max()].index[0]}")
# ---->         return data
# ---->     return data

# ----> # ----> using for loop for removing columns with vif more than 5 iteratively
# ----> for i in range(4):
# ---->     x = vif_remover(x)
# ----> print(x.describe())
######################################################################################################################

# ----> now splitiing data to test and train 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# ----> feature scalling

# ----> apply standard scaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# ----> creating model for prediction 
lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)

# ----> predicting the test set
y_pred = lr.predict(x_test)

# ----> checking metrics

# ----> accuracy score
print(accuracy_score(y_test,y_pred))

# ----> confusion matrix
print(confusion_matrix(y_test,y_pred))

# ----> display confusion matrix 
conf_dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred))
conf_dis.plot(cmap="viridis")
plt.show()