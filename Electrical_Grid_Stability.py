
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the Dataset


df=pd.read_csv("grid_data.csv")
df.head(10)


# # Exploratory Data Analysis

# summary of a DataFrame
df.info()



# checking the presence of missing values in each column
df.isnull().sum()



# descriptive statistics
df.describe()



# Show the counts of observations in each category.
print(df['stabf'].value_counts())

sns.set_style('whitegrid')
sns.countplot(x='stabf',data=df, palette='YlGnBu_r')



# distribution of observations in column 'stab'.
plt.figure(figsize=(8,4))
sns.distplot(df['stab'], color='r')





# correlation heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), annot=True)


# # Train Test Split



X = df.drop(['stab', 'stabf'],axis=1)
y = df['stab']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# # Scaling



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Build the Model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout


X_train.shape


# In[13]:


model = Sequential()
model.add(Dense(12, activation='elu'))
model.add(Dropout(0.5))

model.add(Dense(3,activation='elu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)



model.fit(x=X_train,y=y_train.values,
          validation_split=0.1,
          batch_size=32,epochs=100, callbacks=[early_stop])


# In[16]:


# save the model
model.save('Electrical_Grid_Stability.h5')


# In[17]:


model.summary()


# In[18]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# # Model Evaluation

# In[19]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
predictions = model.predict(X_test)


# In[20]:


predictions


# In[21]:


mean_absolute_error(y_test,predictions)


# In[22]:


mean_squared_error(y_test,predictions)


# In[23]:


np.sqrt(mean_squared_error(y_test,predictions))


# In[24]:


def foo(t1, t2, t3, t4, p1, p2, p3, p4, g1, g2, g3, g4):
    X=pd.DataFrame(data=np.array([[t1, t2, t3, t4, p1, p2, p3, p4, g1, g2, g3, g4]]))
    X_test = scaler.transform(X)
    prediction = model.predict(X_test)
    print(prediction)
    if prediction>=0:
        return "Oops! the system is linearly unstable."
    else:
        return "Great! the system is stable."


# In[25]:


foo(4.689852, 4.007747, 1.478573, 3.733787, 4.041300, -1.410344, -1.238204, -1.392751, 0.269708, 0.250364, 0.164941, 0.482439)


# In[26]:


foo(2.042954, 8.514335, 8.173809, 5.466635, 3.783797, -1.639912,-0.662469, -1.481417, 0.154129, 0.944486, 0.053225, 0.499109)


# In[27]:


foo(6.530527, 6.781790, 4.349695, 8.673138, 3.492807, -1.390285, -1.532193, -0.570329, 0.073056, 0.505441, 0.378761, 0.942631)


# In[28]:


foo(3.392299, 1.274827, 2.954947, 6.894759, 4.349512, -1.663661, -0.952437, -1.733414, 0.502079, 0.567242, 0.285880, 0.366120)

