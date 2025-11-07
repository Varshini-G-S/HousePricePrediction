import pandas as pd
df = pd.read_csv("Housing.csv")
df.head()
df.describe()
df.info()
df.isna().sum()
list1 =['	mainroad','	guestroom','basement','	hotwaterheating','airconditioning','prefarea'] 
df.columns = df.columns.str.strip()
list1 = [col.strip() for col in list1]
df[list1] = df[list1].replace({'yes': 1, 'no': 0})
df.head()
df['furnishingstatus'].nunique()
df.tail()
df['furnishingstatus'] = df['furnishingstatus'].replace({'unfurnished':2,'semi-furnished':1,'furnished':0})
df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
list2 = ['price','area']
df[list2] = scaler.fit_transform(df[list2])
df.head()
df.dtypes
corr = df.corr()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()
df.hist(figsize=(10,10),bins=10)
plt.show()
x = df.drop('price',axis=1)
y = df['price']
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)
from sklearn.metrics import r2_score
lr_accuracy = r2_score(y_test,y_predict)*100
lr_accuracy
# new_data: DataFrame with exactly the feature columns you will feed to the model (except 'price')
new_data = pd.DataFrame({
    'area': [3500],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad': [1],
    'guestroom': [0],
    'basement': [1],
    'hotwaterheating': [0],
    'airconditioning': [1],
    'parking': [2],
    'prefarea': [1],
    'furnishingstatus': [1]
})

# Create a temporary DataFrame with the same columns (and order) that scaler was fitted on:
# scaler was fit on ['price', 'area']  -> so we must provide both in the same order
temp = pd.DataFrame({
    'price': [0] * len(new_data),   # dummy placeholder(s)
    'area' : new_data['area'].values
})

# Transform and copy back only the scaled 'area' column
scaled_temp = scaler.transform(temp)   # returns array with shape (n_samples, 2)
new_data['area'] = scaled_temp[:, 1]   # take the scaled area (column index 1)

# Now predict
predicted_price = lr.predict(new_data)
print("Predicted (scaled) price:", predicted_price)






