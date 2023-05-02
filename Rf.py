#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.ensemble import RandomForestClassifier


# In[2]:


st.title('Predicting Mushroom Type using Random Forest')


# In[3]:


st.sidebar.header('Input Parameters')


# In[4]:


def user_input_features():
    Cap_Shape= st.sidebar.selectbox('cap-shape',('5','2','3','0','4','1' ))
    Cap_Surface= st.sidebar.selectbox('cap-surface',('3','2','0','1'))
    Cap_Color= st.sidebar.selectbox('cap-color',('4','3','2','9','8','0','5','1','7','6'))
    Bruises= st.sidebar.selectbox('bruises',('0','1'))
    Odor= st.sidebar.selectbox('odor',('5','2','8','7','0','3','6','1','4'))
    Gill_Spacing=st.sidebar.selectbox('gill-spacing',('0','1'))
    Gill_Size=st.sidebar.selectbox('gill-size',('0','1'))
    Gill_Color=st.sidebar.selectbox('gill-color',('0','7','10','5','2','3','9','4','1','11','6','8'))
    Stalk_Shape= st.sidebar.selectbox('stalk-shape',('1','0'))
    Stalk_Root= st.sidebar.selectbox('stalk-root',('1','0','3','2','4'))
    Stalk_Surface_Above_Ring= st.sidebar.selectbox('stalk-surface-above-ring',('2','1','0','3'))
    Stalk_Surface_Below_Ring= st.sidebar.selectbox('stalk-surface-below-ring',('2','1','0','3'))
    Stalk_Color_Above_Ring = st.sidebar.selectbox('stalk-color-above-ring',('0','1','2','3','4','5','6','7','8'))
    Stalk_Color_Below_Ring = st.sidebar.selectbox('stalk-color-below-ring',('0','1','2','3','4','5','6','7','8'))
    Ring_Number= st.sidebar.selectbox('ring-number',('0','1','2'))
    Ring_Type= st.sidebar.selectbox('ring-type',('0','1','2','3','4'))
    Spore_Print_Color= st.sidebar.selectbox('spore_print_color',('0','1','2','3','4','5','6','7','8'))
    Population= st.sidebar.selectbox('population',('0','1','2','3','4','5'))
    Habitat= st.sidebar.selectbox('habitat',('0','1','2','3','4','5','6'))
    data= {'cap-shape':Cap_Shape, 'cap-surface':Cap_Surface, 'cap-color':Cap_Color, 'bruises':Bruises, 'odor':Odor,
       'gill-spacing':Gill_Spacing, 'gill-size':Gill_Size, 'gill-color':Gill_Color, 'stalk-shape':Stalk_Shape, 'stalk-root':Stalk_Root,
       'stalk-surface-above-ring':Stalk_Surface_Above_Ring, 'stalk-surface-below-ring':Stalk_Surface_Below_Ring,
       'stalk-color-above-ring':Stalk_Color_Above_Ring  , 'stalk-color-below-ring':Stalk_Color_Below_Ring , 'ring-number':Ring_Number,
       'ring-type':Ring_Type, 'spore-print-color':Spore_Print_Color, 'population':Population, 'habitat':Habitat}
    features= pd.DataFrame(data,index=[0])
    
    return features

  


# In[5]:


df= user_input_features()
st.subheader('User Input Parameters')


# In[6]:


table= pd.read_csv('mushrooms.csv')


# In[7]:


table=table.drop(['veil-type','gill-attachment','veil-color'],axis=1)


# In[8]:


x= table.drop(['class'],axis=1)
y= table['class']


# In[9]:


X=x.apply(LabelEncoder().fit_transform)

le= LabelEncoder()
Y=le.fit_transform(y)


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)


# In[11]:


model=RandomForestClassifier(max_features='sqrt',n_estimators=10)


# In[12]:


model.fit(x_train,y_train)


# In[13]:


frame_1= pd.DataFrame({'cap-shape':['bell=b','conical=c','convex=x','flat=f', 'knobbed=k','sunken=s'],'code':[0,1,5,2,3,4]})
frame_2= pd.DataFrame({'cap-surface': ['fibrous=f','grooves=g','scaly=y','smooth=s'],'code':[0,1,3,2]})
frame_3= pd.DataFrame({'cap-color':['brown=n','buff=b','cinnamon=c','gray=g','green=r','pink=p','purple=u','red=e','white=w','yellow=y'],'code':[4,0,1,3,6,5,7,2,8,9]})
frame_4= pd.DataFrame({'bruises':['present=t','no=f'],'code':[1,0]})
frame_5= pd.DataFrame({'odor': ['almond=a','anise=l','creosote=c','fishy=y','foul=f','musty=m','none=n','pungent=p','spicy=s'],'code':[0,3,1,8,2,4,5,6,7]})
frame_6=pd.DataFrame({'gill-spacing': ['close=c','crowded=w','distant=d'],'code':[0,2,1]})
frame_7= pd.DataFrame({'gill-size':['broad=b','narrow=n'],'code':[0,1]})
frame_8= pd.DataFrame({'gill-color': ['black=k','brown=n','buff=b','chocolate=h','gray=g','green=r','orange=o','pink=p','purple=u','red=e','white=w','yellow=y'],'code':[4,5,0,3,2,8,6,7,9,1,10,11]})                                  
frame_9 = pd.DataFrame({'stalk-shape': ['enlarging=e','tapering=t'],'code':[0,1]}) 
frame_10 = pd.DataFrame({'stalk-root': ['bulbous=b','club=c','cup=u','equal=e','rhizomorphs=z','rooted=r','missing=?'],'code':[1,2,5,3,6,4,0]})
frame_11 = pd.DataFrame({'stalk-surface-above-ring': ['fibrous=f','scaly=y','silky=k','smooth=s'],'code':[0,3,1,2]})
frame_12 = pd.DataFrame({'stalk-surface-below-ring': ['fibrous=f','scaly=y','silky=k','smooth=s'],'code':[0,3,1,2]})
frame_13 = pd.DataFrame({'stalk-color-above-ring':['brown=n','buff=b','cinnamon=c','gray=g','orange=o','pink=p','red=e','white=w','yellow=y'],'code':[4,0,1,3,5,6,2,7,8]})
frame_14 = pd.DataFrame({'stalk-color-below-ring':['brown=n','buff=b','cinnamon=c','gray=g','orange=o','pink=p','red=e','white=w','yellow=y'],'code':[4,0,1,3,5,6,2,7,8]})
frame_15 = pd.DataFrame({'ring-number': ['none=n','one=o','two=t'],'code':[0,1,2]})
frame_16= pd.DataFrame({'ring-type': ['cobwebby=c','evanescent=e','flaring=f','large=l','none=n','pendant=p','sheathing=s','zone=z'],'code':[0,1,2,3,4,5,6,7]})
frame_17= pd.DataFrame({'spore-print-color':['black=k','brown=n','buff=b','chocolate=h','green=r','orange=o','purple=u','white=w','yellow=y'],'code':[2,3,0,1,5,4,6,7,8]})
frame_18= pd.DataFrame({'population': ['abundant=a','clustered=c','numerous=n','scattered=s','several=v','solitary=y'],'code':[0,1,2,3,4,5]})
frame_19= pd.DataFrame({'habitat': ['grasses=g','leaves=l','meadows=m','paths=p','urban=u','waste=w','woods=d'],'code':[1,2,3,4,5,6,0]})


# In[14]:


st.dataframe(frame_1, use_container_width=True)
st.dataframe(frame_2, use_container_width=True)
st.dataframe(frame_3, use_container_width=True)
st.dataframe(frame_4, use_container_width=True)
st.dataframe(frame_5, use_container_width=True)
st.dataframe(frame_6, use_container_width=True)
st.dataframe(frame_7, use_container_width=True)
st.dataframe(frame_8, use_container_width=True)
st.dataframe(frame_9, use_container_width=True)
st.dataframe(frame_10, use_container_width=True)
st.dataframe(frame_11, use_container_width=True)
st.dataframe(frame_12, use_container_width=True)
st.dataframe(frame_13, use_container_width=True)
st.dataframe(frame_14, use_container_width=True)
st.dataframe(frame_15, use_container_width=True)
st.dataframe(frame_16, use_container_width=True)
st.dataframe(frame_17, use_container_width=True)
st.dataframe(frame_18, use_container_width=True)
st.dataframe(frame_19, use_container_width=True)


# In[15]:


predict=model.predict(df)


# In[16]:


st.subheader('Predicted Type')


# In[19]:


st.write('Edible'if predict==0 else 'Poisonous')


# In[18]:





# In[ ]:




