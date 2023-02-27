import streamlit as st
from matplotlib import image
import pandas as pd
import plotly.express as px
import seaborn as sb
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


st.set_option('deprecation.showPyplotGlobalUse', False)

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "images", "iris.jpg")
DATA_PATH = os.path.join(dir_of_interest, "data", "iris.csv")

st.title("Dashboard - Iris Data")

img = image.imread(IMAGE_PATH)
st.image(img)

df = pd.read_csv(DATA_PATH)

choice = st.selectbox('SELECT',['-','DATA SET','PLOTS','DATA_ANALYSIS','MODEL'])
if choice == 'DATA SET':st.dataframe(df)
if choice == 'PLOTS':
    plots = st.selectbox('PLOTS',['-','HIST','BOX','DIST'])
    if plots == 'HIST':
        fig = plt.figure(figsize=(10, 4))
        cl = st.selectbox('select the label',df['Species'].unique())
        fea = st.selectbox('select the feature',df.columns[0:4])
        sb.histplot(df[df['Species'] == cl], x=fea)
        st.pyplot(fig)
    if plots =='BOX':
        fig1 = plt.figure(figsize=(10, 4))
        fea1 = st.selectbox('select the feature',df.columns[0:4])
        sb.boxplot(df[fea1])
        st.pyplot(fig1)
    if plots == 'DIST':
        fig2 = plt.figure(figsize=(10, 4))
        fea2 = st.selectbox('select the feature',df.columns[0:4])
        sb.distplot(df[fea2])
        st.pyplot(fig2)
if choice == 'DATA_ANALYSIS':
    da = st.selectbox('DATA_ANALYSIS',['-','check best feature that separate labels','outliers','null_values','descirptive','check for balanced dataset'])
    if da == 'check best feature that separate labels':
        st.subheader('See the distribution of the features') 
        a = st.checkbox(df.columns[0])
        b = st.checkbox(df.columns[1])
        c = st.checkbox(df.columns[2])
        d = st.checkbox(df.columns[3])
        fig = plt.figure(figsize=(10, 4))
        if a==True:
            sb.distplot(df[df.columns[0]])
        if b==True:
            sb.distplot(df[df.columns[1]])
        if c==True:
            sb.distplot(df[df.columns[2]])
        if d==True:
            sb.distplot(df[df.columns[3]])
        st.pyplot(fig)
    if da == 'outliers':
        fig1 = plt.figure(figsize=(10, 4))
        fea1 = st.selectbox('select the feature',df.columns[0:4])
        sb.boxplot(df[fea1])
        st.pyplot(fig1)
    if da == 'null_values':
        st.write(df.isnull().sum())
    if da == 'descirptive':
        st.write(df.describe())
    if da == 'check for balanced dataset':
        st.write(df['Species'].value_counts())
if choice == 'MODEL':
    features_variables = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
    labels = df[["Species"]]
    mod = st.selectbox('select the feature',['-','knn'])
    if mod == 'knn':
        n = st.number_input('select the train size',min_value=0.5,max_value=1.0,step=0.01)
        x_train,x_test,y_train,y_test = train_test_split(features_variables,labels,train_size=n)
        x_train_cv,x_cv,y_train_cv,y_cv = train_test_split(x_train,y_train,train_size=n)  
        k= st.number_input('give the range of k to get best hyper parameter',min_value=2,key='int',step=1)
        train_error = []
        cv_error = []
        k_value = []
        for i in range(1,k,2):
            knn = KNeighborsClassifier(n_neighbors=i)
            model = knn.fit(x_train_cv,y_train_cv)
            y_predicted = model.predict(x_train_cv)
            acu = accuracy_score(y_train_cv,y_predicted)
            train_error.append(1-acu)
        for i in range(1,k,2):
            knn = KNeighborsClassifier(n_neighbors=i)
            model = knn.fit(x_train_cv,y_train_cv)
            y_predicted = model.predict(x_cv)
            acu = accuracy_score(y_cv,y_predicted)
            k_value.append(i)
            cv_error.append(1-acu)
        fig = plt.figure(figsize=(10, 4))
        plt.plot(k_value,train_error)
        plt.xticks(k_value)
        plt.plot(k_value,cv_error)
        plt.xticks(k_value)
        st.pyplot(fig)
        k_final = st.number_input('choose the best hyper parameter from the graph',min_value=2,step=1)
        knn = KNeighborsClassifier(n_neighbors=k_final)
        model = knn.fit(x_train_cv,y_train_cv)
        y_predicted = model.predict(x_test)
        acu = accuracy_score(y_test,y_predicted)
        st.write('The accuarcy of the model is',acu)
st.caption('MORE FEATURES WILL BE ADDED IN THE FUTURE')