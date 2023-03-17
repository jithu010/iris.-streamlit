import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import shap
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse',False)

st.markdown("<h1 style='align-text:center'> IRIS Length Prediction </h1>",unsafe_allow_html=True)
st.write('This app predicts  __Iris Features__')

st.write('---')

iris = datasets.load_iris()
X=pd.DataFrame(iris.data,columns=iris.feature_names)
Y=pd.DataFrame(iris.target,columns=['target'])

iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_data['target']=pd.DataFrame(iris.target)
st.dataframe(iris_data)

chart_select=st.sidebar.selectbox(
    label="Selct type of Graphs",
    options=['scatter','line','histogram','box'])




numeric_columns= list(iris_data.select_dtypes(['float','int']).columns)


def select_graph(chart_select):
    st.sidebar.subheader("Visualization  Settings")
    try:
        x_values=st.sidebar.selectbox("X axis",options=numeric_columns)
        y_values=st.sidebar.selectbox("y axis",options=numeric_columns)
        if chart_select=='scatter':
            plot=px.scatter(data_frame=iris_data,x=x_values,y=y_values)
        elif chart_select=='histogram':
            plot = px.histogram(data_frame=iris_data, x=x_values, y=y_values)
        elif chart_select=='line':
            plot = px.line(data_frame=iris_data, x=x_values, y=y_values)
        elif chart_select=='box':
            plot = px.box(data_frame=iris_data, x=x_values, y=y_values)
        st.write(plot)
    except Exception as er:
        st.write(er)

select_graph(chart_select)


st.sidebar.subheader("Input Parameters")
def user_input_features():
    sepal_l=st.sidebar.slider("Sepal_Length",float(X['sepal length (cm)'].min()),float(X['sepal length (cm)'].max()),float(X['sepal length (cm)'].mean()))
    petal_l=st.sidebar.slider("Petal_Length",float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()),float(X['petal length (cm)'].mean()))
    petal_w=st.sidebar.slider("Petal_width",float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()),float(X['petal width (cm)'].mean()))
    sepal_w=st.sidebar.slider("Sepal_width",float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()),float(X['sepal width (cm)'].mean()))
    data={'sepal_lenth (cm)':sepal_l,'petal_length (cm)':petal_l,'petal_width (cm)':petal_w,'sepal_width (cm)':sepal_w}
    features=pd.DataFrame(data=data,index=[0])
    return features

st.write("Input Features")
st.dataframe(user_input_features())

model=RandomForestRegressor()
model.fit(X,Y)

prediction=model.predict(X)

st.subheader("Predicted Values")
st.write(prediction)

explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(X)
if st.button('Show Feature Importance'):
    st.header('Feature Importance')
    plt.title('Feature Importance based on Shap Values')
    shap.summary_plot(shap_values,X)
    st.pyplot(bbox_inches='tight')
    st.write('---')
    plt.title('Feature Importance based on Shap Values')
    shap.summary_plot(shap_values,X,plot_type='bar')
    st.pyplot(bbox_inches='tight')











