import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import shap
from sklearn.cluster import KMeans
from zipfile import ZipFile
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA


@st.cache
def load_data():
	
    z = ZipFile('train_sample_30mskoriginal.zip') 
    data = pd.read_csv(z.open('train_sample_30mskoriginal.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
    data = data.drop('Unnamed: 0', axis=1)
    z = ZipFile('train_sample_30m.zip')
    sample = pd.read_csv(z.open('train_sample_30m.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
    X_sample = sample.iloc[:, :-1]
    
    description = pd.read_csv('HomeCredit_columns_description.csv', 
    				usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')
    				
    target = data[['TARGET']] # target = data.iloc[:, -1:]
    
    return data, sample, X_sample, target, description


def load_model():
        '''loading the trained model'''
        pickle_in = open('LGBMClassifier_best_customscore.pkl', 'rb') 
        model = pickle.load(pickle_in)
        return model


@st.cache(allow_output_mutation=True)
def load_knn(sample):
    knn = knn_training(sample)
    return knn


@st.cache
def load_gen_info(data):
    list_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]


    nb_credits = list_infos[0]
    mean_revenue = list_infos[1]
    mean_credits = list_infos[2]

    targets = data.TARGET.value_counts()

    return nb_credits, mean_revenue, mean_credits, targets
    
    
def client_identity(data, id):
    data_client = data[data.index == int(id)]
    return data_client    


@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"]/365), 2)
    return data_age


@st.cache
def load_income_population(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income


@st.cache
def load_prediction(sample, id, model):
    X=sample.iloc[:, :-1]
    score = model.predict_proba(X[X.index == int(id)])[:,1]
    return score


@st.cache
def knn_training(sample):
    knn = KMeans(n_clusters=2, random_state=7).fit(sample)
    return knn 
    

@st.cache
def load_kmeans(sample, id, model):
    index = sample[sample.index == int(id)].index.values
    index = index[0]
    data_client = pd.DataFrame(X_sample.loc[sample.index, :])
    df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data], axis=1)
    return df_neighbors.iloc[:,1:].sample(10)


@st.cache
def load_pca_proj(sample, id, model):
    index_proj = sample[sample.index == int(id)].index.values
    index_proj = index_proj[0]
    data_client_proj = pd.DataFrame(X_sample.loc[sample.index, :])
    knn1 = knn_training(data_client_proj)
    df_neighbors_proj = pd.DataFrame(knn1.fit_predict(data_client_proj), index=data_client_proj.index)
    df_neighbors_proj = pd.concat([df_neighbors_proj, sample], axis=1)
    return df_neighbors_proj.iloc[:,1:].sample(10)


# @st.cache
def perform_pca(data_import):
    pca, pca_data, pca_cols, num_data = pca_maker(data_import)
    pca_1 = pca_cols[0]
    pca_2 = pca_cols[1]
    pca_data['TARGET'] = data_import['TARGET']
    pca_data['TARGET'] = pca_data['TARGET'].astype(str)
    fig = px.scatter(data_frame=pca_data, x=pca_1, y=pca_2, template="simple_white", color='TARGET', width=500, height=500)
    fig.update_traces(marker_size=10)
    return scatter_column.plotly_chart(fig)
    
    
# @st.cache
def pca_maker(data_import):
    numerical_columns_list = []
    categorical_columns_list = []

    for i in data_import.columns:
        if data_import[i].dtype == np.dtype("float64") or data_import[i].dtype == np.dtype("int64"):
            numerical_columns_list.append(data_import[i])
        else:
            categorical_columns_list.append(data_import[i])

    numerical_data = pd.concat(numerical_columns_list, axis=1)
    # categorical_data = pd.concat(categorical_columns_list, axis=1)

    numerical_data = numerical_data.apply(lambda x: x.fillna(np.mean(x)))

    pca1 = PCA(random_state=7)

    pca_data = pca1.fit_transform(data_import.iloc[:, :-1])

    pca_data = pd.DataFrame(pca_data, index=data_import.index)
        
    new_column_names = ["PCA_" + str(i) for i in range(1, len(pca_data.columns) + 1)]

    column_mapper = dict(zip(list(pca_data.columns), new_column_names))

    pca_data = pca_data.rename(columns=column_mapper)

    output = pd.concat([data_import, pca_data], axis=1)

    return pca1, output, new_column_names, list(numerical_data.columns)
    

# Global feature importance
@st.cache
def get_model_varimportance(model, train_columns, max_vars=10):
    var_imp_df = pd.DataFrame([train_columns, model.feature_importances_]).T
    var_imp_df.columns = ['feature_name', 'var_importance']
    var_imp_df.sort_values(by='var_importance', ascending=False, inplace=True)
    var_imp_df = var_imp_df.iloc[0:max_vars] 
    return var_imp_df


# Loading data
data, sample, X_sample, target, description = load_data()
id_client = sample.index.values
model = load_model()




#******************************************
# MAIN -- title
#******************************************

# Title display
html_temp = """
<div style="background-color: LightSeaGreen; padding:5px; border-radius:10px">
	<h1 style="color: white; text-align:center">Credit Allocation Dashboard</h1>
</div>
    """
st.markdown(html_temp, unsafe_allow_html=True)




#*******************************************
# Displaying informations on the sidebar
#*******************************************

# Loading selectbox
# st.sidebar.header('Pick client ID')
# chk_id = st.sidebar.selectbox('', id_client)
chk_id = st.sidebar.selectbox('Pick client ID', id_client)

# Loading general informations
nb_credits, mean_revenue, mean_credits, targets = load_gen_info(data)

# Number of loans for clients in study
st.sidebar.markdown("<u>Total number of loans in our sample :</u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

# Average income
st.sidebar.markdown("<u>Average income ($US) :</u>", unsafe_allow_html=True)
st.sidebar.text(int(mean_revenue))

# AMT CREDIT
st.sidebar.markdown("<u>Average loan amount ($US) :</u>", unsafe_allow_html=True)
st.sidebar.text(int(mean_credits))

# Labels explanation
st.sidebar.markdown("<u>Labels explanation :</u>", unsafe_allow_html=True)
st.sidebar.text('TARGET 1 : "Defaulted"')
st.sidebar.text('TARGET 0 : "Reimbursed"')





#******************************************
# MAIN -- all the rest
#******************************************

'''-------------------------------------------------------------------------------------------------'''



# Customer prediction display
prediction = load_prediction(sample, chk_id, model)
st.header("Default probability : {:.0f} %".format(round(float(prediction)*100, 2)))

infos_client = client_identity(data, chk_id)
st.write('Client label :', infos_client['TARGET'])
'''-------------------------------------------------------------------------------------------------'''



# Gauge chart
plot_bgcolor = "white"

quadrant_colors = [plot_bgcolor, "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"] 
quadrant_text = ["", "<b>Very high</b>", "<b>High</b>", "<b>Medium</b>", "<b>Low</b>", "<b>Very low</b>"]
n_quadrants = len(quadrant_colors) - 1

current_value = round(float(prediction)*100, 0)
min_value = 0
max_value = 50
hand_length = np.sqrt(2) / 4.7
hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

fig = go.Figure(
        data=[
          go.Pie(
            values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
            rotation=90,
            hole=0.6,
            marker_colors=quadrant_colors,
            text=quadrant_text,
            textinfo="text",
            hoverinfo="skip")],
        layout=
          go.Layout(
            showlegend=False,
            margin=dict(b=0,t=10,l=10,r=10),
            width=700,
            height=600,
            paper_bgcolor=plot_bgcolor,
            annotations=[
              go.layout.Annotation(
                text=f"<b>Default probability :</b><br>{current_value} %",
                x=0.5, xanchor="center", xref="paper",
                y=0.25, yanchor="bottom", yref="paper",
                showarrow=False)],
            shapes=[
              go.layout.Shape(
                type="circle",
                x0=0.48, x1=0.52,
                y0=0.48, y1=0.52,
                fillcolor="#333",
                line_color="#333"),
              go.layout.Shape(
                type="line",
                x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                line=dict(color="#333", width=4))]))
                
st.plotly_chart(fig)                           
'''-------------------------------------------------------------------------------------------------'''



# Displaying customer information : gender, age, family status, Nb of hildren etc.
st.subheader('Customer general informations')
# Display Customer ID from Sidebar
st.write('Customer selected :', chk_id)

# Age informations
# infos_client = client_identity(data, chk_id)
st.write("Gender : ", infos_client["CODE_GENDER"].values[0])
st.write("Age : {:.0f} years old".format(int(infos_client["DAYS_BIRTH"]/365)))
st.write("Family status : ", infos_client["NAME_FAMILY_STATUS"].values[0])
st.write("Number of children : {:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
'''-------------------------------------------------------------------------------------------------'''



# Financial informations   
st.subheader("Customer financial informations ($US)")
st.write("Income total : {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
st.write("Credit amount : {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
st.write("Credit annuities : {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
st.write("Amount of property for credit : {:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
'''-------------------------------------------------------------------------------------------------'''
   
   
   
# Feature importance
st.subheader("Customer report")
if st.checkbox("Show (Hide) customer #{:.0f} feature importance".format(chk_id)):
   st.markdown("<h5 style='text-align: center;'>Customer feature importance</h5>", unsafe_allow_html=True)
   shap.initjs()
   X = sample.iloc[:, :-1] # X = sample.loc[:, sample.columns != 'TARGET']
   X = X[X.index == chk_id]
   number = st.slider("Chose number of features up to 10", 0, 10, 5)

   fig, ax = plt.subplots(figsize=(7,7))
   explainer = shap.TreeExplainer(load_model())
   shap_values = explainer.shap_values(X)
   shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(3, 3))
   st.pyplot(fig)

else:
   st.markdown("<i>…</i>", unsafe_allow_html=True)
'''-------------------------------------------------------------------------------------------------'''


# Global feature importance
if st.checkbox("Show(Hide) global feature imortance") :
      st.markdown("<h5 style='text-align: center;'>Global feature importance</h5>", unsafe_allow_html=True)
      feature_importance = get_model_varimportance(model, sample.iloc[:, :-1].columns) # sample.columns
      fig = px.bar(feature_importance, x='var_importance', y='feature_name', orientation='h')
      st.plotly_chart(fig)
            
      # Customer data
      st.markdown("<u>Customer data with most global important features</u>", unsafe_allow_html=True)
      st.write(client_identity(data[feature_importance.feature_name], chk_id))
      
      # Feature description
      st.markdown("<u>Feature description</u>", unsafe_allow_html=True)
      list_features = description.index.to_list()
      feature = st.selectbox('You can type first letters of the feature for proposition', list_features)
      st.table(description.loc[description.index == feature][:1]) 

else:
   st.markdown("<i>…</i>", unsafe_allow_html=True)
'''-------------------------------------------------------------------------------------------------'''
   
   
   
# Closest customer projections

st.subheader('Projection of closest customers')
X_proj = load_pca_proj(sample, chk_id, model)
scatter_column, settings_column = st.columns((309, 1))
perform_pca(X_proj)
'''-------------------------------------------------------------------------------------------------'''



# Distribution plots : age & income 
   
  
# Age distribution plot --> OK
if st.checkbox("Enable (Disable) showing disribution plots"):
   st.subheader('Distribution plots')
   data_age = load_age_population(data)
   data_age = pd.DataFrame(data_age)
   data_age_labels = pd.concat([data_age, data[['TARGET']]], axis=1)
   st.markdown("<h5 style='text-align: center;'>Customer age</h5>", unsafe_allow_html=True)
   fig = px.histogram(data_age_labels, color='TARGET', nbins=50, histnorm='probability', barmode="overlay")
   fig.update_layout(xaxis_title="Age (Years)", yaxis_title="Count_norm")
   fig.add_vline(x=int(infos_client["DAYS_BIRTH"].values / 365), line_width=5, line_dash="dash", line_color="green")
   st.plotly_chart(fig, theme=None) 
   
   
   # Income distribution plot
   st.markdown("<h5 style='text-align: center;'>Customer income</h5>", unsafe_allow_html=True)
   data_income = load_income_population(data)
   data_income = pd.DataFrame(data_income)
   data_income = pd.concat([data_income, data[['TARGET']]], axis=1)
   fig = px.histogram(data_income, color='TARGET', nbins=15, histnorm='probability', barmode="overlay") 
   fig.update_layout(xaxis_title="Income ($US)", yaxis_title="Count_norm")
   fig.add_vline(x=int(infos_client["AMT_INCOME_TOTAL"].values[0]), line_width=5, line_dash="dash", line_color="green")
   st.plotly_chart(fig, theme=None)
'''-------------------------------------------------------------------------------------------------''' 


   
# PieChart Defaulted/Reimbursed
if st.checkbox("Enable (Disable) customer repartition"):
   st.subheader('Repartition of customers by labels')
   st.markdown("<h5 style='text-align: center;'>0 -- reimbursed | 1 -- defaulted</h5>", unsafe_allow_html=True)
   fig = px.pie(data, names='TARGET') #, color_discrete_sequence=px.colors.sequential.RdBu)
   st.plotly_chart(fig, theme=None)
