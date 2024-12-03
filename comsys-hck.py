import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
print(df.head())
print(df.columns.value_counts())


print(df.shape)

print(df.info())

print(df.isnull().sum()) # 73 null values in country
#74 null values at beginning of 2020/21
#30 null values at beginning of 2021/22
#5 null values at beginning of 2022/23  Rest all OK---> 4 colums with null values

print("No of unique countries/categories in countries column is :"+" "+str(len(df.Country.unique()))) # A lot of categories
print(df.Country.value_counts())  #MAX -- Spain, France, Italy, Germany, England
# df.Country = np.where(df.Country=="Spain",1,0)
# df.Country = df.Country.astype(str)
# One Hot Encoding need to be done to handle such categories

print("No of unique names in countries column is :"+" "+str(len(df.Name.unique())))

count = df.Country
print(count)
df = df.drop(["Name","Country"], axis=1)
col_names = df.columns

print(col_names)

# print(df.Country.value_counts())


# for col_name in col_names:
#     if(df[col_name].dtypes=='int64' or df[col_name].dtypes=='float64'):
#         plt.boxplot(df[col_name])
#         plt.xlabel(col_name)
#         plt.ylabel('count')
#         plt.show()

# Outlier removal

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)] # axis = 1 --> column wise
print(df.shape)

# for col_name in col_names:
#     if(df[col_name].dtypes=='int64' or df[col_name].dtypes=='float64'):
#         plt.boxplot(df[col_name])
#         plt.xlabel(col_name)
#         plt.ylabel('count')
#         plt.show()

df['Country'] = count


for i in col_names:
    if df[i].dtypes =='object':
        #print('ob')
        df[i] = df[i].fillna(df[i].mode()[0])
    else:
        #print('num')
        df[i] = df[i].fillna(df[i].median())

df.Country = df.Country.fillna(df.Country.mode()[0])
print(df.isnull().sum()) #Null values treated

print(df.shape)

# for col_name in col_names:
#     if(df[col_name].dtypes=='int64' or df[col_name].dtypes=='float64'):
#         sns.kdeplot(data=df,x=col_name)
#         plt.xlabel(col_name)
#         plt.ylabel('count')
#         plt.show()

print(df.Country.value_counts())

#--------Loads of Categories so encoding the top categories only

counts = df['Country'].value_counts() # This is a Series object
threshold = 5
repl = counts[counts<=threshold]

others = list(repl.keys()) # All the columns below the threshold are in this list
for i in others:
    df.Country = df.Country.str.replace(i,"Others")
print(df.Country)

# One Hot Encoding the country column categories --because the features have no ordinal relationship
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Country"]=le.fit_transform(df["Country"]) #--> Takes 2D array as input so converting single column values to 2D array

# SUCCESSFULLY ENCODED VALUES
df["Country"] = df["Country"].astype(float)



print(df.head())

# for i in df.columns:
#     sns.scatterplot(x=df[i], y = df["Value at beginning of 2023/24 season"])
#     plt.show()

#Normalizing all values
# for i in df.columns:
#     df[i] = (df[i] - df[i].mean()) / df[i].std()
# print("----------------------------------AFTER NORMALIZATION AND CLEANING----------------------------")
# print(df.head())


# for col_name in col_names:
#     if(df[col_name].dtypes=='int64' or df[col_name].dtypes=='float64'):
#         sns.kdeplot(data=df,x=col_name)
#         plt.xlabel(col_name)
#         plt.ylabel('count')
#         plt.show()

#
# sns.kdeplot(data=df,x="Country")
# plt.show()

print(df.info())

# Checking the variance inflation factor between the columns

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

print(col_list) #Independent features list

X = df[col_list]


# VIF Checking

from statsmodels.stats.outliers_influence import variance_inflation_factor
# col_list = []
# for col in data.columns:
#     if ((data[col].dtype != 'object') & (col != 'y') ):
#         col_list.append(col)

vif_data = pd.DataFrame() # Creating a new data frame
vif_data["feature"] = X.columns # adding a column- feature which will contain all the column names
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


df = df.drop("Percentage of Passes Completed",axis=1)

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
#
# df = df.drop("Attempted Passes", axis=1)
print("\n")


col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
#
df = df.drop("Attacking options created", axis =1)
#
#
col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)

# df = df.drop("Age", axis =1)
# df = df.drop("Country", axis=1)

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
#
df = df.drop("Value at beginning of 2021/22 season", axis =1)

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
#
df = df.drop("Touches in attacking penalty area", axis =1)

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
#
df = df.drop("Shots", axis =1)
col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)

# df = df.drop("Blocks", axis =1)

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)

# df = df.drop("Progressive Carries", axis=1)
col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
#
# df = df.drop("Tackles", axis =1)

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
# #
df = df.drop("Expected Goal Contributions", axis =1)

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
#
# df = df.drop("Open Play Expected Goals", axis =1)

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
#
df = df.drop(["Interceptions","id"], axis=1)
col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)
# df = df.drop("Progressive Passes", axis=1)

col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Value at beginning of 2023/24 season') ):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
for i in range(len(X.columns))]
print(vif_data)



#
# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.70)
# IQR = Q3 - Q1
# df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)] # axis = 1 --> column wise
# print(df.shape)

# Q1 = df.Age.quantile(0.25)
# Q3 = df.Age.quantile(0.75)
# IQR = Q3 - Q1
# df = df[~((df.Age < (Q1 - 1.5 * IQR)) |(df.Age > (Q3 + 1.5 * IQR)))] # axis = 1 --> column wise
# print(df.shape)


# for i in df.columns:
#     sns.boxplot(df[i])
#     plt.xlabel(i)
#     plt.ylabel("Count")
#     plt.show()



print(df.corr())

# for i in df.columns:
#     sns.kdeplot(data=df,x=i)
#     plt.show()

data_ind = vif_data.feature

df_ind = df[data_ind]

df_dep = df['Value at beginning of 2023/24 season']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_ind, df_dep, test_size=0.20, random_state=100)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

test_predict = lr.predict(x_test)


sns.scatterplot(x=y_test,y=test_predict)
plt.show()

from sklearn.metrics import *
r2_on_test=r2_score(y_test, test_predict)
print(r2_on_test)

from sklearn import metrics
print("Mean Absolute Error",metrics.mean_absolute_error(y_test,test_predict))
print("Mean Squared Error",np.sqrt(metrics.mean_squared_error(y_test,test_predict)))

error_pred=pd.DataFrame(columns=['Actual_data','Prediction_data'])
error_pred['Actual_data'] = y_test
error_pred['Prediction_data'] = test_predict

print(error_pred)



df2= pd.read_csv("test.csv")

print(df2.isnull().sum())

print(df2.head())
print("No of unique countries/categories in countries column is :"+" "+str(len(df2.Country.unique())))  # A lot of categories

col_names2 = list(df2.columns)
print(col_names2)

# for col_name in col_names2:
#     if(df2[col_name].dtypes=='int64' or df2[col_name].dtypes=='float64'):
#         sns.boxplot(df2[col_name])
#         plt.xlabel(col_name)
#         plt.ylabel('count')
#         plt.show()

for i in col_names2:
    if df2[i].dtypes =='object':
        #print('ob')
        df2[i] = df2[i].fillna(df2[i].mode()[0])
    else:
        #print('num')
        df2[i] = df2[i].fillna(df2[i].median())

print(df2.isnull().sum())
le2 = LabelEncoder()
df2["Country"]=le2.fit_transform(df2["Country"]) #--> Takes 2D array as input so converting single column values to 2D array

# SUCCESSFULLY ENCODED VALUES OF TEST DATASET COUNTRY COLUMN
df2["Country"] = df2["Country"].astype(float)

count2 = df2.Country
count_id = df2.id

df2 = df2.drop(["id","Interceptions","Expected Goal Contributions","Touches in attacking penalty area","Value at beginning of 2021/22 season","Attacking options created","Shots","Country","Percentage of Passes Completed"], axis =1)

df2['Country'] = count2

print(df.columns)
print(df2.columns)


# print(df2.columns)
# print("\n")
# print(df.columns)


X_test = df2.iloc[:,:]
print(X_test)
final_predictions = lr.predict(X_test)
print(final_predictions)

final_csv = pd.DataFrame(columns=['id','label'])
final_csv['id'] = count_id
final_csv['label'] = final_predictions

print(final_csv)

final_csv.to_csv("predictions.csv", encoding='utf-8', index=False)










































