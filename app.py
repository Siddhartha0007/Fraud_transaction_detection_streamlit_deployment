# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Wed Aug  3 19:42:01 2022
# 
# 
# @author: Siddhartha-Sarkar
# """
# =============================================================================

    
 ###  Import Libreries  
import streamlit as st
from streamlit_option_menu import option_menu
st.set_option('deprecation.showPyplotGlobalUse', False)
import contractions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from collections import  Counter
import inflect
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
import os
#for model-building
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm
#for model accuracy
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
#for visualization
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import pickle
from joblib import dump, load
import joblib
# Utils
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import sys
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error 
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from plotly import tools
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
import plotly.figure_factory as ff
import cufflinks as cf
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
# Setting seabon style
sns.set_style(style='darkgrid')
import scipy
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox, probplot, norm
from scipy.special import inv_boxcox
import random
import datetime
import math
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
## Hyperopt modules
#from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
#from functools import partial
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import re
import sys
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import re
import sys
from yellowbrick.classifier import PrecisionRecallCurve
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import plot_importance, to_graphviz
from xgboost import plot_tree
#For Model Acuracy
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import plot_importance, to_graphviz #
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
# EDA Pkgs
import pandas as pd 
import codecs
from pandas_profiling import ProfileReport 
# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

# Custome Component Fxn
import sweetviz as sv 


le_encoder=LabelEncoder()
###############################################Data Processing###########################
data=pd.read_csv("new_balanced_data.csv")
loaded_model=pickle.load(open("Xgb_classifer_model_intelligence.pkl","rb"))


def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)


def user_input_features():
    
    amount = st.sidebar.number_input("enter the amount")
    oldbalanceOrg = st.sidebar.number_input("enter the oldbalance of the sender or source")
    newbalanceOrig = st.sidebar.number_input("enter the newbalance of the sender or source")
    oldbalanceDest = st.sidebar.number_input("enter the oldbalance of the receiver")
    newbalanceDest = st.sidebar.number_input("enter the newbalance of the receiver")
    typeid= st.sidebar.selectbox("Transaction Modes:",['PAYMENT', 'DEBIT', 'CASH_IN', 'TRANSFER', 'CASH_OUT'] )
   
    
    
    
    data = {'amount':amount,
            'oldbalanceOrg':oldbalanceOrg,
            'newbalanceOrig':newbalanceOrig,
            'oldbalanceDest':oldbalanceDest,
            'newbalanceDest':newbalanceDest,
            'typeid':typeid
            }

    features = pd.DataFrame(data,index = [0])
    
    return features
        


###############################################Exploratory Data Analysis###############################################

#For Label Analysis
def label_analysis():
    st.write("Different types of transactions modes")
    
    image1= Image.open("im1.png")
    st.image(image1,use_column_width=False)
    #st.write("Random under Sampling")
    
    image1= Image.open("im2.png")
    st.image(image1,use_column_width=False)
    st.write("fraud and not fraud transaction distributions with percentage")
    
    image1= Image.open("im3.png")
    st.image(image1,use_column_width=False)
    st.write("Different types of transactions modes under not fraud data")
    
    image1= Image.open("im4.png")
    st.image(image1,use_column_width=False)
    st.write("Different types of transactions modes under fraud data")
    
    image1= Image.open("im5.png")
    st.image(image1,use_column_width=False)
    st.write("Continous variables distributions")
    
    image1= Image.open("im6.png")
    st.image(image1,use_column_width=False)
    st.write("Continous variables distributions using box plot before outliers treatments")
    
    image1= Image.open("im7.png")
    st.image(image1,use_column_width=False)
    st.write("Continous variables distributions using box plot")
    
    image1= Image.open("im8.png")
    st.image(image1,use_column_width=False)
    st.write("Continous variables distributions before winsorisation")
    
    image1= Image.open("im9.png")
    st.image(image1,use_column_width=False)
    st.write("Continous variables distributions after winsorisation")
    
    image1= Image.open("im10.png")
    st.image(image1,use_column_width=False)
# =============================================================================
#     
#      
#     
#     def plot9():
#         fig = px.treemap(data.loc[:,:], path=[ 'type','Fraud_Id'], values='amount', color='Fraud_Id')
#         fig.show()
#         return fig
#     p9=plot9()
#     st.write("Multi level Tree plot for various types of transactions")
#     st.plotly_chart(p9)
# =============================================================================
  
def label_analysis1():
    st.write("Multi level SunBurst Plots")
    
    image1= Image.open("im12.png")
    st.image(image1,use_column_width=False)
    
    st.write("Multi level SunBurst Plots")
    
    image1= Image.open("im13.png")
    st.image(image1,use_column_width=False)
    
    
    


def label_analysis2():
    st.write("Random under Sampling")
    
    image1= Image.open("random_under.png")
    st.image(image1,use_column_width=False)
    
    
    st.write("TomKlinks Sampling")
    
    image2= Image.open("tomklinks.png")
    st.image(image2,use_column_width=False)
    st.write("TomKlimks + NearMiss Sampling")
    
    image3= Image.open("tomklink+nearmiss.png")
    st.image(image3,use_column_width=False)
    st.write("Result Of the Under Sampling")
    
    image4= Image.open("undersampling_result.png")
    st.image(image4,use_column_width=False)

def label_analysis3():
    st.write("Feature importance")
    
    image1= Image.open("feature_imp.png")
    st.image(image1,use_column_width=False)
    
    
    st.write("Feature Selection Using Mutual Information")
    
    image2= Image.open("mutual_feature_imp.png")
    st.image(image2,use_column_width=False)

def label_analysis4():
    df_sample1 =data.describe(include='all').round(2).T
    colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
    fig =  ff.create_table(df_sample1, colorscale=colorscale)
    fig.show()
    return st.plotly_chart(fig)

def label_analysis5():
    def plot13():
        corr_matrix = data.corr()
        f,ax = plt.subplots(figsize=(14,6))
        sns.heatmap(corr_matrix,annot=True,linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax,cmap='rainbow')
        plt.show()
         
    p13=plot13()
    st.write("Correlation Matrix")
    st.pyplot(p13)
    

    
    def plot14():
        corr = data.corrwith(data['Fraud_Id'],method='spearman').reset_index()
        corr.columns = ['Index','Correlations']
        corr = corr.set_index('Index')
        corr = corr.sort_values(by=['Correlations'], ascending = False).head(10)
        plt.figure(figsize=(10, 8))
        fig = sns.heatmap(corr, annot=True, fmt="g", cmap='coolwarm', linewidths=0.4, linecolor='red')
        plt.title("Correlation of Variables with Class", fontsize=20)
        plt.show()
         
    p14=plot14()
    st.write(" Correlation Matrix")
    st.pyplot(p14)
    
    def plot15():
        import seaborn as sns
        corr_matrix = data.corr()
        sns.set(rc = {'figure.figsize':(12,7)}) # handle size of thr figure 
        #mask = np.zeros_like(corr_matrix)
        mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
        #mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            sns.heatmap(corr_matrix, mask=mask ,annot=True, square=True,linewidths=0.5, 
                  fmt= '.2f',cmap='coolwarm');
            
          
         
    p15=plot15()
    st.write("Correlation Matrix")
    st.pyplot(p15)



def get_data_class():
    data=pd.read_csv("new_balanced_data.csv")
    X=data.drop(['Fraud_Id'],axis=1)
    y=data[['Fraud_Id']]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    #Standardizing the numerical columns
    col_names=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
    features_train = X_train[col_names]
    features_test = X_test[col_names]
    scaler = StandardScaler().fit(features_train.values)
    features_train = scaler.transform(features_train.values)
    features_test = scaler.transform(features_test.values)
    X_train[col_names] = features_train
    X_test[col_names] =features_test
    return X_train,X_test,y_train,y_test  
    

############################################### Model Learning ###############################################
#For Precision Recall Curve
def PRCurve(model):
    X_train,X_test,y_train,y_test=get_data_class()
    prc = PrecisionRecallCurve(model)
    prc.fit(X_train, y_train)
    avg_prc = prc.score(X_test, y_test)
    plt.legend(labels = ['Precision Recall Curve',"AP=%.3f"%avg_prc], loc = 'lower right', prop={'size': 14})
    plt.xlabel(xlabel = 'Recall', size = 14)
    plt.ylabel(ylabel = 'Precision', size = 14)
    plt.title(label = 'Precision Recall Curve', size = 16)
    
  

   
#Model Random Forest Regressor
def randomforest_classifier(data):
    X_train,X_test,y_train,y_test=get_data_class()
    rf =RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=4,
                            max_features='auto',random_state=None,class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_train_predict=rf.predict(X_train)
    rf_prediction = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    st.write("Random Forest Classification Train Accuracy: {}%".format(round(rf.score(X_train,y_train)*100,2)))
    st.write("Random Forest Classification Test Accuracy: {}%".format(round(rf.score(X_test,y_test)*100,2)))
    rf_cm = confusion_matrix(y_test, rf_prediction)
    st.write("Classification Report:Train data\n")
    st.markdown(classification_report(y_train,rf_train_predict))
    st.write("Classification Report:Test data\n")
    st.markdown(classification_report(y_test, rf_prediction))
    st.write("Logloss:\n",log_loss(y_test, rf.predict_proba(X_test)))
    st.write('Confusion Matrix \n', rf_cm)
    #Precision Recall Curve
    plt.figure(figsize = [10, 8])
    PRCurve(rf)
    st.pyplot()
    plot_confusion_matrix(rf, X_test,y_test)
    st.pyplot()
    plot_roc_curve(rf, X_test, y_test)
    st.pyplot()
#Model XGBoost  Regressor
def XGboost_classifier(data):
    X_train,X_test,y_train,y_test=get_data_class()
    xgb_classifer= XGBClassifier(n_estimators=200,max_depth=6,booster="gbtree",learning_rate=0.005)
    xgb_classifer.fit(X_train,y_train)
    xgb_train_predict=xgb_classifer.predict(X_train)
    xgb_prediction = xgb_classifer.predict(X_test)
    xgb_score = xgb_classifer.score(X_test, y_test)
    st.write("XGB Classification Train Accuracy: {}%".format(round(xgb_classifer.score(X_train,y_train)*100,2)))
    st.write("XGB Classification Test Accuracy: {}%".format(round(xgb_classifer.score(X_test,y_test)*100,2)))
    xgb_classifer_cm = confusion_matrix(y_test, xgb_prediction)
    st.write("Classification Report:Train data\n")
    st.write(classification_report(y_train, xgb_train_predict))
    st.write("------------------------------------------------------\n")
    st.write("Classification Report:Test data\n")
    st.write(classification_report(y_test, xgb_prediction))
    st.write("Logloss:\n",log_loss(y_test, xgb_classifer.predict_proba(X_test)))
    st.write('Confusion Matrix \n', xgb_classifer_cm)
    #Precision Recall Curve
    plt.figure(figsize = [10, 8])
    PRCurve(xgb_classifer)
    st.pyplot()
    plot_confusion_matrix(xgb_classifer, X_test,y_test)
    st.pyplot()
    plot_roc_curve(xgb_classifer, X_test, y_test)
    st.pyplot()




def predict_func():
    df=user_input_features()
    st.write(df)
    df["typeid"]=le_encoder.fit_transform(df["typeid"])
    #X_train,X_test,y_train,y_test=get_data_class()
    #xgb_classifer= XGBClassifier(n_estimators=200,max_depth=6,booster="gbtree",learning_rate=0.005)
    #xgb_classifer.fit(X_train,y_train)
    
    
    html_temp = """
     <div style="background-color:royalblue;padding:10px;border-radius:10px">
     <h1 style="color:white;text-align:center;">After Entering the inputs press predict Button</h1>
         </div>  """
    components.html(html_temp)
    #st.write("After Entering the inputs press predict Button")
    if st.button("Predict"):
        y_test_pred=loaded_model.predict(df)
        return y_test_pred
        
    
    
        
###############################################Streamlit Main###############################################

def main():
    # set page title
    
    
            
    # 2. horizontal menu with custom style
    selected = option_menu(menu_title=None, options=["Home", "Projects","Report" ,"About"], icons=["house", "book","app-indicator","envelope"],  menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": " #f08080 "},"icon": {"color": "blue", "font-size": "25px"}, "nav-link": {"font-size": "25px","text-align": "left","margin": "0px","--hover-color": "#eee", },           "nav-link-selected": {"background-color": "green"},},)
    
    #horizontal Home selected
    if selected == "Home":
        st.title(f"You have selected {selected}")
        
        image= Image.open("home_img.jpg")
        st.image(image,use_column_width=True)
            
        st.sidebar.title("Home")        
        with st.sidebar:
            image= Image.open("Home1.png")
            add_image=st.image(image,use_column_width=True)
              
        st.sidebar.write("Video Of the Project")
        st.sidebar.video("Fraud Detection.mp4")
        with st.sidebar:
            image= Image.open("Home.png")
            add_image=st.image(image,use_column_width=True)
        st.balloons()
        #st.title('Fraudulent Transaction Detection Using Machine Learning')
        #st.video("https://youtu.be/O73OPzkUlR0")
        
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;"> Fraudulent Transaction Detection Using Machine Learning</h1>
		</div>  """
        
		
        components.html(html_temp)
        
        html_temp11 = """
		 What is fraudulent transactions?
         
         A fraudulent transaction is the unauthorized use of an individual’s accounts or payment information.
         Fraudulent transactions can result in the victim’s loss of funds, personal property, or personal information
         The fraudulent transaction is one of the most serious threats to online security nowadays.
         Artificial Intelligence is vital for financial risk control in the cloud environment.
         Many studies attempted to explore methods for online transaction fraud detection; however,
         the existing methods are insufficient to conduct detection with high precision.
         Fraud prevention is the implementation of a strategy to detect fraudulent transactions or banking actions
         and prevent these actions from causing financial damage and the reputation of the client and the financial 
         institution.
         There are always financial frauds and They can happen in virtual and physical ways.
         So the investment in security has been increasing. Keep your business safe from online 
         payment fraud There is no guaranteed method for payment fraud prevention. 
         By taking certain precautions, however, you can minimize the damage they cause and make sure your 
         business has the best chance to thrive despite them.
        
		  """
        
		
        st.write(html_temp11)
        def plot11():
            import plotly.graph_objects as go

            values = [['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest','Fraud_Id','isFlaggedFraud'], #1st col
                     ["maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation)",
                        "CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER",
                                 "amount of the transaction in local currency",
                                 "customer who started the transaction",
                                   " initial balance before the transaction",
                                    "new balance after the transaction",
                           " customer who is the recipient of the transaction",
                            "initial balance recipient before the transaction.Note that there is not information for customers that start with M (Merchants)",
                             "new balance recipient after the transaction. Note that there is not",
                                 "This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system",
                                      " The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfermore than 200000 in a single transaction"
                                      ]]


            fig = go.Figure(data=[go.Table(
            columnorder = [1,2],
            columnwidth = [100,400],
            header = dict(
            values = [['<b>Columns<br>of the Dataset</b>'],
                  ['<b>DESCRIPTION</b>']],
            line_color='darkslategray',
            fill_color='royalblue',
            align=['left','center'],
            font=dict(color='white', size=12),
            height=40
  ),
            cells=dict(
            values=values,
            line_color='darkslategray',
            fill=dict(color=['green', 'pink']),
            align=['left', 'center'],
            font_size=12,
            height=20)
    )
    ])
       
            return fig   
        p11=plot11()
        st.write("About Dataset")
        st.plotly_chart(p11)
        
        def plot12():
            import plotly.figure_factory as ff
            df_sample = data.iloc[0:10,0:9]
            colorscale = [[0, '#F08080'],[.5, '#6495ED'],[1, '#9FE2BF']]
            fig =  ff.create_table(df_sample, colorscale=colorscale,)
            fig.show()
            return fig
        p12=plot12()
        st.write("Data Table")
        st.plotly_chart(p12)
        ### features
# =============================================================================
#         image= Image.open("word-image-20.png")
#         st.image(image,use_column_width=True)
#         st.header('Features')
# =============================================================================
        def header(url):
            st.markdown(f'<p style="background-color:royalblue ;color:white;font-size:20px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
        html_temp111 = """
		The fraudulent transaction is one of the most serious threats to online security nowadays. 
        Artificial Intelligence is vital for financial risk control in the cloud environment. Many studies attempted to explore methods for online transaction fraud detection; however, the existing methods are insufficient to conduct detection with high precision.
        Fraud prevention is the implementation of a strategy to detect fraudulent transactions or banking actions and prevent these actions from causing financial damage and the reputation of the client and the financial institution
        Therefore the Aim of this project is to build a model which is able to predict whether a transaction is fraudulent or not.
         .
        """
        header(html_temp111)
        st.markdown("""
                #### Tasks Perform by the app:
                + App covers the most basic Machine Learning task of  Analysis, Correlation between variables,project report.
                + Machine Learning on different Machine Learning Algorithms, building different models and lastly  prediction.
                
                """)
                
    #Horizontal About selected
    if selected == "About":
        st.title(f"You have selected {selected}")
        
        st.sidebar.title("About")
        with st.sidebar:
            image= Image.open("About-Us-PNG-Isolated-Photo.png")
            add_image=st.image(image,use_column_width=True)        
        
        #st.image('iidt_logo_137.png',use_column_width=True)
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">This is a Fraudulent Transaction Detection Project</h1>
		</div>  """
        
		
        components.html(html_temp)
        st.markdown("<h2 style='text-align: center;'> This Project aims to Detect <br>Fraudulent Transaction using Machine Learning</h2>", unsafe_allow_html=True)

        st.sidebar.markdown("""
                    #### + Project Done By :        
                    #### @Author Mr. Siddhartha Sarkar
                    #### Group-1 Members: Mr. Siddhartha Sarkar,Sarathchandra Karnati,Rahul Singh)
        
                    """)
        st.snow()
        image2= Image.open("About1.jpg")
        st.image(image2,use_column_width=True)
        st.sidebar.markdown("[ Visit To Github Repositories](.git)")
    #Horizontal Project_Report selected
    if selected == "Report":
        
        st.title("You have selected Profile Report")
        st.sidebar.title("Project_Profile_Report")
        
        with st.sidebar:
            image= Image.open("report_project.png")
            add_image=st.image(image,use_column_width=True)
        
        st.balloons()    
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Simple EDA App with Streamlit Components</h1>
		</div>  """
        
		
        components.html(html_temp)
        html_temp1 = """
			<style>
			* {box-sizing: border-box}
			body {font-family: Verdana, sans-serif; margin:0}
			.mySlides {display: none}
			img {vertical-align: middle;}
			/* Slideshow container */
			.slideshow-container {
			  max-width: 1500px;
			  position: relative;
			  margin: auto;
			}
			/* Next & previous buttons */
			.prev, .next {
			  cursor: pointer;
			  position: absolute;
			  top: 50%;
			  width: auto;
			  padding: 16px;
			  margin-top: -22px;
			  color: white;
			  font-weight: bold;
			  font-size: 18px;
			  transition: 0.6s ease;
			  border-radius: 0 3px 3px 0;
			  user-select: none;
			}
			/* Position the "next button" to the right */
			.next {
			  right: 0;
			  border-radius: 3px 0 0 3px;
			}
			/* On hover, add a black background color with a little bit see-through */
			.prev:hover, .next:hover {
			  background-color: rgba(0,0,0,0.8);
			}
			/* Caption text */
			.text {
			  color: #f2f2f2;
			  font-size: 15px;
			  padding: 8px 12px;
			  position: absolute;
			  bottom: 8px;
			  width: 100%;
			  text-align: center;
			}
			/* Number text (1/3 etc) */
			.numbertext {
			  color: #f2f2f2;
			  font-size: 12px;
			  padding: 8px 12px;
			  position: absolute;
			  top: 0;
			}
			/* The dots/bullets/indicators */
			.dot {
			  cursor: pointer;
			  height: 15px;
			  width: 15px;
			  margin: 0 2px;
			  background-color: #bbb;
			  border-radius: 50%;
			  display: inline-block;
			  transition: background-color 0.6s ease;
			}
			.active, .dot:hover {
			  background-color: #717171;
			}
			/* Fading animation */
			.fade {
			  -webkit-animation-name: fade;
			  -webkit-animation-duration: 1.5s;
			  animation-name: fade;
			  animation-duration: 1.5s;
			}
			@-webkit-keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			@keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			/* On smaller screens, decrease text size */
			@media only screen and (max-width: 300px) {
			  .prev, .next,.text {font-size: 11px}
			}
			</style>
			</head>
			<body>
			<div class="slideshow-container">
			<div class="mySlides fade">
			  <div class="numbertext">1 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_nature_wide.jpg" style="width:100%">
			  <div class="text">Caption Text</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">2 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_snow_wide.jpg" style="width:100%">
			  <div class="text">Caption Two</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">3 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_mountains_wide.jpg" style="width:100%">
			  <div class="text">Caption Three</div>
			</div>
			<a class="prev" onclick="plusSlides(-1)">&#10094;</a>
			<a class="next" onclick="plusSlides(1)">&#10095;</a>
			</div>
			<br>
			<div style="text-align:center">
			  <span class="dot" onclick="currentSlide(1)"></span> 
			  <span class="dot" onclick="currentSlide(2)"></span> 
			  <span class="dot" onclick="currentSlide(3)"></span> 
			</div>
			<script>
			var slideIndex = 1;
			showSlides(slideIndex);
			function plusSlides(n) {
			  showSlides(slideIndex += n);
			}
			function currentSlide(n) {
			  showSlides(slideIndex = n);
			}
			function showSlides(n) {
			  var i;
			  var slides = document.getElementsByClassName("mySlides");
			  var dots = document.getElementsByClassName("dot");
			  if (n > slides.length) {slideIndex = 1}    
			  if (n < 1) {slideIndex = slides.length}
			  for (i = 0; i < slides.length; i++) {
			      slides[i].style.display = "none";  
			  }
			  for (i = 0; i < dots.length; i++) {
			      dots[i].className = dots[i].className.replace(" active", "");
			  }
			  slides[slideIndex-1].style.display = "block";  
			  dots[slideIndex-1].className += " active";
			}
			</script>
			"""
        components.html(html_temp1)    
        menu = ['None',"Sweetviz","Pandas Profile"]
        choice = st.sidebar.radio("Menu",menu)
        if choice == 'None':
            st.markdown("""
                        #### Kindly select from left Menu.
                       # """)
        elif choice == "Pandas Profile":
            st.subheader("Automated EDA with Pandas Profile")
            #data_file= st.file_uploader("Upload CSV",type=['csv'])
            df = data
            st.table(df.head(10))
            profile= ProfileReport(df)
            st_profile_report(profile)
        elif choice == "Sweetviz":
            st.subheader("Automated EDA with Sweetviz")
            #data_file = st.file_uploader("Upload CSV",type=['csv'])
            df =data
            st.dataframe(df.head(10))
            if st.button("Generate Sweetviz Report"):

				# Normal Workflow
                report = sv.analyze(df)
                report.show_html()
                st_display_sweetviz("SWEETVIZ_REPORT.html")  
    			
		       
                
    			
            
		
        
    #Horizontal Project selected
    if selected == "Projects":
            st.title(f"You have selected {selected}")
            with st.sidebar:
                #image= Image.open("project_side.jpg")
                #add_image=st.image(image,use_column_width=True)
                image= Image.open("project_hgff.png")
                add_image=st.image(image,use_column_width=True)
            import time

                
                          
            image2= Image.open("project_img.jpg")
            st.image(image2,use_column_width=True)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.sidebar.title("Navigation")
            menu_list1 = ['Exploratory Data Analysis',"Prediction With Machine Learning"]
            menu_Pre_Exp = st.sidebar.radio("Menu For Prediction & Exploratoriy", menu_list1)
            
            #EDA On Document File
            if menu_Pre_Exp == 'Exploratory Data Analysis' and selected == "Projects":
                    st.title('Exploratory Data Analysis')

                    
                    
                    menu_list2 = ['None', 'Basic_Statistics','Basic_Plots','Multi_level_SunBurst_plots','Under_Sampling','Feature importance','Correlation_Matrices']
                    menu_Exp = st.sidebar.radio("Menu EDA", menu_list2)

                    
                    if menu_Exp == 'None':
                        st.markdown("""
                                    #### Kindly select from left Menu.
                                   # """)
                    
                    elif menu_Exp == 'Basic_Plots':
                        label_analysis()
                    elif menu_Exp == 'Multi_level_SunBurst_plots':
                        label_analysis1()
                    elif menu_Exp == 'Under_Sampling':
                        label_analysis2()
                    elif menu_Exp == 'Feature importance':
                        label_analysis3()
                    elif menu_Exp == 'Basic_Statistics':
                        label_analysis4()   
                    elif menu_Exp == 'Correlation_Matrices': 
                        label_analysis5()

            elif menu_Pre_Exp == "Prediction With Machine Learning" and selected == "Projects":
                    st.title('Prediction With Machine Learning')
                    
                    menu_list3 = ['Checking ML Method And Various Matrices' ,'Prediction' ]
                    menu_Pre = st.radio("Menu Prediction", menu_list3)
                    
                    #Checking ML Method And Accuracy
                    if menu_Pre == 'Checking ML Method And Various Matrices':
                            st.title('Checking Accuracy  And Various Matrices On Different Algorithms')
                            #dataframe=data_func(data)
                            
                            if st.checkbox("View data"):
                                st.write(data)
                            model = st.selectbox("ML Method",[ 'XGB Classifier', 'Random Forest Classifier'])
                            #vector= st.selectbox("Vector Method",[ 'CountVectorizer' , 'TF-IDF'])

                            if st.button('Analyze'):
                                #Logistic Regression 
                                #if model=='Logistic Regression':
                                    #logistic_regression(get_data_class(final_data))
                                    #st.write(data)
                                

                                #XGB Classifier
                                if model=='XGB Classifier':
                                    XGboost_classifier(get_data_class())
                                                                           
                                    #st.write(data)
                               
                                
                                
                                #Random Forest Classifier & CountVectorizer
                                elif model=='Random Forest Classifier':
                                    randomforest_classifier(get_data_class())
                                    #st.write(data)
                    #Checking ML Method And Accuracy
                    #elif menu_Pre == 'Checking Regression Method And Accuracy':
                            #st.title('Checking Accuracy On Different Algorithms')
                            #dataframe=data_func(data)
                            
                            #if st.checkbox("View data"):
                                #st.write(data)
                            #model = st.selectbox("ML Method",['XGboost_regressor', 'Random Forest Regressor'])
                            #vector= st.selectbox("Vector Method",[ 'CountVectorizer' , 'TF-IDF'])

                           # if st.button('Analyze'):
                                #Logistic Regression 
                                #if model=='XGboost_regressor':
                                   # XGboost_regressor(get_data_reg(dataframe))
                                    
                                

                                #XGB Classifier & CountVectorizer
                                #elif model=='Random Forest Regressor':
                                    #randomforest_regressor(get_data_reg(dataframe))                                    
                                
                              
                                          
                    elif menu_Pre == 'Prediction':
                        st.title('Prediction')
                        with st.spinner('Wait for it...'):
                            time.sleep(5)
    
                           
                        #df= user_input_features()
                        
                        result_pred = predict_func()
                        if (result_pred==0):
                            st.write("Your Transaction is not Fraud")
                            image1= Image.open("im15.png")
                            st.image(image1,use_column_width=False)
                        else:
                            st.write("Your Transaction is Fraud")
                            image1= Image.open("im14.png")
                            st.image(image1,use_column_width=False)
                            
                        
                        st.success('Done!')       
                        #st.success('The Transaction is --> {}'.format(result_pred))
                        
                            
                                

                                                      
if __name__=='__main__':
    main()            
            
            

