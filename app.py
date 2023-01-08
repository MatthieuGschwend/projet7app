import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import plotly.graph_objects as go
from plotly.graph_objs import *
import numpy as np
import json
import shap
import xgboost
import pickle
import streamlit.components.v1 as components
import base64
import math



def plot_categorical_variables_pie(data, column_name, plot_defaulter = True, hole = 0):
    '''
    Function to plot categorical variables Pie Plots
    
    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        plot_defaulter: bool
            Whether to plot the Pie Plot for Defaulters or not
        hole: int, default = 0
            Radius of hole to be cut out from Pie Chart
    '''
    if plot_defaulter:
        cols = 2
        specs = [[{'type' : 'domain'}, {'type' : 'domain'}]]
        titles = [f'Distribution of {column_name} for all Targets', f'Percentage of Defaulters for each category of {column_name}']
    else:
        cols = 1
        specs = [[{'type': 'domain'}]]
        titles = [f'Distribution of {column_name} for all Targets']
        
    values_categorical = data[column_name].value_counts()
    labels_categorical = values_categorical.index
    
    fig = make_subplots(rows = 1, cols = cols, 
                       specs = specs, 
                       subplot_titles = titles)
    
    #plotting overall distribution of category
    fig.add_trace(go.Pie(values = values_categorical, labels = labels_categorical, hole = hole, 
                         textinfo = 'label+percent', textposition = 'inside'), row = 1, col = 1)
    
    #plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = data[column_name][data.TARGET == 1].value_counts() * 100 / data[column_name].value_counts()
        percentage_defaulter_per_category.dropna(inplace = True)
        percentage_defaulter_per_category = percentage_defaulter_per_category.round(2)
        
        fig.add_trace(go.Pie(values = percentage_defaulter_per_category, labels = percentage_defaulter_per_category.index, 
                             hole = hole, textinfo = 'label+value', hoverinfo = 'label+value'), row = 1, col = 2)
        
    fig.update_layout(title = f'Distribution of {column_name}')
    fig.show()
    
    
def plot_categorical_variables_bar(data, column_name, figsize = (18,6), percentage_display = True, plot_defaulter = True, rotation = 0, horizontal_adjust = 0, fontsize_percent = 'xx-small'):
    '''
    Function to plot Categorical Variables Bar Plots
    
    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default = (18,6)
            Size of the figure to be plotted
        percentage_display: bool, default = True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default = 0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default = 'xx-small'
            Fontsize for percentage Display
        
    '''
    print(f"Total Number of unique categories of {column_name} = {len(data[column_name].unique())}")
    
    plt.figure(figsize = figsize, tight_layout = False)
    sns.set(style = 'whitegrid', font_scale = 1.2)
    
    #plotting overall distribution of category
    plt.subplot(1,2,1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending = False)
    ax = sns.barplot(x = data_to_plot.index, y = data_to_plot, palette = 'Set1')
    
    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax.patches:
            ax.text(p.get_x() + horizontal_adjust, p.get_height() + 0.005 * total_datapoints, '{:1.02f}%'.format(p.get_height() * 100 / total_datapoints), fontsize = fontsize_percent)
        
    plt.xlabel(column_name, labelpad = 10)
    plt.title(f'Distribution of {column_name}', pad = 20)
    plt.xticks(rotation = rotation)
    plt.ylabel('Counts')
    
    #plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.TARGET == 1].value_counts() * 100 / data[column_name].value_counts()).dropna().sort_values(ascending = False)

        plt.subplot(1,2,2)
        sns.barplot(x = percentage_defaulter_per_category.index, y = percentage_defaulter_per_category, palette = 'Set2')
        plt.ylabel('Percentage of Defaulter per category')
        plt.xlabel(column_name, labelpad = 10)
        plt.xticks(rotation = rotation)
        plt.title(f'Percentage of Defaulters for each category of {column_name}', pad = 20)
    st.pyplot(plt)
    
    
def plot_continuous_variables(data, column_name, plots = ['distplot', 'CDF', 'box', 'violin'], scale_limits = None, figsize = (20,8), histogram = True, log_scale = False):
    '''
    Function to plot continuous variables distribution
    
    Inputs:
        data: DataFrame
            The DataFrame from which to plot.
        column_name: str
            Column's name whose distribution is to be plotted.
        plots: list, default = ['distplot', 'CDF', box', 'violin']
            List of plots to plot for Continuous Variable.
        scale_limits: tuple (left, right), default = None
            To control the limits of values to be plotted in case of outliers.
        figsize: tuple, default = (20,8)
            Size of the figure to be plotted.
        histogram: bool, default = True
            Whether to plot histogram along with distplot or not.
        log_scale: bool, default = False
            Whether to use log-scale for variables with outlying points.
    '''
    data_to_plot = data.copy()
    if scale_limits:
        #taking only the data within the specified limits
        data_to_plot[column_name] = data[column_name][(data[column_name] > scale_limits[0]) & (data[column_name] < scale_limits[1])]

    number_of_subplots = len(plots)
    plt.figure(figsize = figsize)
    sns.set_style('whitegrid')
    
    for i, ele in enumerate(plots):
        plt.subplot(1, number_of_subplots, i + 1)
        plt.subplots_adjust(wspace=0.25)
        
        if ele == 'CDF':
            #making the percentile DataFrame for both positive and negative Class Labels
            percentile_values_0 = data_to_plot[data_to_plot.TARGET == 0][[column_name]].dropna().sort_values(by = column_name)
            percentile_values_0['Percentile'] = [ele / (len(percentile_values_0)-1) for ele in range(len(percentile_values_0))]
            
            percentile_values_1 = data_to_plot[data_to_plot.TARGET == 1][[column_name]].dropna().sort_values(by = column_name)
            percentile_values_1['Percentile'] = [ele / (len(percentile_values_1)-1) for ele in range(len(percentile_values_1))]
            
            plt.plot(percentile_values_0[column_name], percentile_values_0['Percentile'], color = 'red', label = 'Non-Defaulters')
            plt.plot(percentile_values_1[column_name], percentile_values_1['Percentile'], color = 'black', label = 'Defaulters')
            plt.xlabel(column_name)
            plt.ylabel('Probability')
            plt.title('CDF of {}'.format(column_name))
            plt.legend(fontsize = 'medium')
            if log_scale:
                plt.xscale('log')
                plt.xlabel(column_name + ' - (log-scale)')
            
        if ele == 'distplot':  
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 0].dropna(),
                         label='Non-Defaulters', hist = False, color='red')
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 1].dropna(),
                         label='Defaulters', hist = False, color='black')
            plt.xlabel(column_name)
            plt.ylabel('Probability Density')
            plt.legend(fontsize='medium')
            plt.title("Dist-Plot of {}".format(column_name))
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{column_name} (log scale)')

        if ele == 'violin':  
            sns.violinplot(x='TARGET', y=column_name, data=data_to_plot)
            plt.title("Violin-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

        if ele == 'box':  
            sns.boxplot(x='TARGET', y=column_name, data=data_to_plot)
            plt.title("Box-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

    st.pyplot(plt)
    
    
def make_jauge(val = 328,
               previous_val = 400,
               objective_value = 370,
               max_value = 500,
               titre = "Speed",
               half = False,
              ):
    
    color_bande1 = "rgb(255, 50, 50)"
    color_bande2 = "rgb(225, 124, 124)"
    color_bande3 = "rgb(142, 210, 61)"
    color_bande4 = "rgb(126, 186, 54)"

    color_delta = 'black'#'rgb(104, 168, 32)'
    tickcolor = "darkblue"
    color_bar = "rgb(128, 111, 3)"#"rgb(86, 198, 233)"
    color_back = "white"
    color_seuil = 'blue'
    color_font = "rgb(128,111,3)"
    paper_bgcolor = "rgba(220, 219, 211,0.5)"
    
    title_size = 24
    
    if half == True:
        title_size = 15
        


    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': titre, 'font': {'size': title_size}},
        delta = {'reference': previous_val, 'increasing': {'color': color_delta}},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 0.5, 'tickcolor': tickcolor},
            'bar': {'color': color_bar},
            'bgcolor': color_back,
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, (max_value)/4], 'color': color_bande1},
                {'range': [(max_value)/4, (max_value)/2], 'color': color_bande2},
                {'range': [(max_value)/2, 3*(max_value)/4], 'color': color_bande3},
                {'range': [3*(max_value)/4, max_value], 'color': color_bande4}],
            'threshold': {
                'line': {'color': "blue", 'width': 10},
                'thickness': 0.90,
                'value': objective_value}}))

    fig.update_layout(paper_bgcolor = paper_bgcolor, font = {'color': "rgb(5, 70, 40)", 'family': "Arial"})
    
    if half == True:
        fig.update_layout(
        autosize=False,
        width=230,
        height=230,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=10,
            pad=0
        ),
        paper_bgcolor = "rgba(220, 219, 211,0)")
        fig.update_yaxes(automargin=True)
    
    
    return fig
    
def score_5(x, seuil):
    min_score = 1 - 2*(1-seuil)
    if x < min_score:
        return 0
    else :
        return 25*(x - min_score)
 

def get_model_explainer(model):
    explainer  = shap.TreeExplainer(model)
    return explainer

def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)
        
 
@st.cache
def get_shap_values_train(explainer,data_train):
    shap_values = explainer.shap_values(data_train)
    return shap_values
    
@st.cache(allow_output_mutation=True)
def read_data_init():
    model =  pickle.load(open('model.pkl','rb'))
    
    data_train = pd.read_csv('df_full_train_streamlit.csv')
    data_test = pd.read_csv('df_full_test_streamlit.csv')  
    
    application_test = pd.read_csv('application_test_streamlit.csv')
    application_train = pd.read_csv('application_train_streamlit.csv')
    
    shap_values_train = np.loadtxt('shap_values_train.txt', dtype=float)
   
    data_train.drop(columns = ['SK_ID_CURR', 'TARGET'], inplace = True)
    #data_test.drop(columns = 'Unnamed: 0', inplace = True)
    data_test.set_index( 'SK_ID_CURR', inplace = True)
    application_train['AMT_INCOME_TOTAL_log'] = np.log10(application_train['AMT_INCOME_TOTAL'])
    application_test['AMT_INCOME_TOTAL_log'] = np.log10(application_test['AMT_INCOME_TOTAL'])
    
    
    return model, data_train, data_test, application_test, application_train, shap_values_train

@st.cache(allow_output_mutation=True, persist = True)
def shap_summary(shap_values_train, data_train):
    fig = plt.figure()
    shap.summary_plot(shap_values_train, data_train, plot_type="bar")
    
    
@st.cache(allow_output_mutation=True, persist = True)
def shap_summary2(shap_values_train, data_train):
    fig = plt.figure()
    shap.summary_plot(shap_values_train, data_train)
    return fig

def cast_score(score_client, seuil):
    if score_client < seuil :
        std_segment = score_client/seuil -1 
    else :
        std_segment = (1/(1-seuil)*score_client + seuil/(seuil-1))
    
    val_sigmo = 5/(1+math.exp(-5*std_segment)) 
    return val_sigmo
    
def interface():
    
    model, data_train, data_test, application_test, application_train, shap_values_train = read_data_init()
    
    
    explainer  = get_model_explainer(model)
    #st.write(data_test)
    
    types_analyse = ["Analyse client", "Interprétation globale du modèle"]
    txt = "Types d'analyses: " 
    mode = st.sidebar.selectbox(txt, types_analyse)
    
    
    
    if mode == "Analyse client":
        seuil_acceptation = 0.92
        
        st.header("Analyse client")
        id_client = st.text_input('Numéro client', 100005)
        info_client = pd.DataFrame(application_test[application_test["SK_ID_CURR"] == int(id_client)])
        info_client_to_request = str(data_test.loc[int(id_client)].values.tolist()).strip('[]')

        st.subheader("Score du client")
        
        col1, col2 = st.columns(2)
        #r = requests.get('http://flaskexemple-env.eba-jdusgpeb.us-east-1.elasticbeanstalk.com/predict')
        #r = json.loads(r.text)['predictions'][0][0]
        #url = 'http://127.0.0.1:5000/predictproba?'
        #r = requests.post(url, params={'feature_array': info_client_to_request})
        
        
        
        url = 'http://flaskexemple-env.eba-jdusgpeb.us-east-1.elasticbeanstalk.com/predictproba'
        r = requests.post(url, params={'feature_array': info_client_to_request})
        
        st.write(r)
        
        score_client = json.loads(r.text)['predictions'][0][0]
        st.write(r)
        #val_5 = score_5(0.95, seuil_acceptation)
        val_5 = cast_score(score_client, seuil_acceptation)
        fig = make_jauge(val = val_5,
               previous_val = None ,
               objective_value = 2.5,
               max_value = 5,
               titre = " "
              )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Explication du score")
        
        
        ligne_client = pd.DataFrame(data_test.loc[int(id_client),:]).T
        shap_values_Model = explainer.shap_values(ligne_client)
        st_shap(shap.force_plot(explainer.expected_value, shap_values_Model, ligne_client))
        #explainer = shap.explainers.Tree(model)
        #shap_values = explainer(data_test)
        #shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],shap_values[0].values[:,0], feature_names=dataframe_dentrainement.columns)
        
        st.subheader("Comparaison du client par rapport aux autres clients")
        application_client = application_test[application_test['SK_ID_CURR'] == id_client]
        variables_analyse = {"NAME_CONTRACT_TYPE" : 'cat',
                            "AMT_INCOME_TOTAL_log" : 'num',
                            "NAME_EDUCATION_TYPE" : 'cat',
                            "NAME_INCOME_TYPE" : 'cat',
                            "EXT_SOURCE_2" : 'num'}

                            
        mode = st.selectbox(txt, variables_analyse.keys()) 
        val_client = info_client[mode].values[0]
        try : 
            val_client = round(val_client,3)
        except :
            pass
        st.metric('Valeur pour ce client :', val_client)
        if variables_analyse[mode] == 'cat':
            plot_categorical_variables_bar(application_train, column_name = mode, rotation = 60, horizontal_adjust = 0.25) 
        else :
            plot_continuous_variables(application_train, mode, plots = ['distplot','box' ] )
    
    if mode == "Interprétation globale du modèle":
        st.header('Analyse globale du modèle')
        def download_model(model):
            output_model = pickle.dumps(model)
            b64 = base64.b64encode(output_model).decode()
            href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Télécharger le modèle </a>'
            st.markdown(href, unsafe_allow_html=True)
        
        st.subheader('Description du modèle')
        st.write("Le modèle utilisé pour faire cette classification est un XGboost" )
        
        download_model(model)
        st.subheader('Influence globale des variables ')
        st.image('global.png')
        #st.pyplot(shap_summary(shap_values_train, data_train))
        #st.pyplot(shap_summary2(shap_values_train, data_train))
        
        
            
    

interface()