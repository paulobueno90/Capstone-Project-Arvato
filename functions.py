# import libraries here; add more as necessary
import numpy as np
from numpy import mean
from numpy import std

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pylab import *


def read_and_standardize_attributes():
    """
    Read Attributes excel files
    
    return:
    attributes: attributes data in pandas format
    attributes_info: attributes information levels file data in pandas format
    """
    attributes = pd.read_excel('./DIAS Attributes - Values 2017.xlsx', header=1, index_col=None)
    attributes_info = pd.read_excel('./DIAS Information Levels - Attributes 2017.xlsx', header=1, index_col=None)
    
    attributes= attributes.reset_index().drop(columns=['index'], axis=1)
    attributes_info = attributes_info.reset_index().drop(columns=['index'], axis=1)
    
    return attributes, attributes_info

def fix_warning_cols(df):
    """
    Function to replace 'X', 'XX' and 'nan' values with -1 in 
    columns 18 and 19 i.e. ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015'] 
    
    """
    cols = ["CAMEO_DEUG_2015", "CAMEO_INTL_2015", 'CAMEO_DEU_2015']
    
    df[cols] = df[cols].replace({"X": np.nan, "XX": np.nan})
    
    try:
        df[cols] = df[cols].astype(float)
    except:
        pass
        
    for column in cols:
        
        values = df[column].unique()
        
        for value in values:
            
            try:
                item = float(value)
                df[cols] = df[cols].replace({value: item})
                
            except:
                pass
            
    return df

def standardize_dataset(df):
    """
    Search for data matching attributes file data and convert unknown to -1
    and data without unknown categorie convert to NaN
    
    input:
    df - pandas dataframe
    output:
    df - standardized with attributes file values
    """
    with open('attributes.pkl', 'rb') as f:
            attributes = pickle.load(f)
    
    df["TITEL_KZ"] = df["TITEL_KZ"].replace(0, np.nan)
    df["TITEL_KZ"] = df["TITEL_KZ"].replace(-1, np.nan)
    
    # List of Attributes with 'unknown' values expected
    columns_with_unknown = attributes[attributes['Meaning'].str.contains('unknown') & (attributes['Value'] == -1)]['Attribute'].unique()

    # List of attributes without 'unknown' values and not numerical values
    without_unknown_or_numeric_attributes = attributes[(~attributes['Attribute'].isin(columns_with_unknown)) 
                                                       & (~attributes['Meaning'].str.contains('numeric'))]['Attribute'].unique().tolist()

    # List of numeric attributes
    numeric_attributes = attributes[(attributes['Meaning'].str.contains('numeric')) 
                                    & (~attributes['Attribute'].isin(columns_with_unknown))]['Attribute'].unique().tolist()
    
    for column in columns_with_unknown:

        attribute_values = attributes[attributes['Attribute'] == column]['Value'].tolist()
        dataframe_column_values = df[column].unique().tolist()

        if not all(item in attribute_values for item in dataframe_column_values):
            not_matches_list = [attribute for attribute in dataframe_column_values if attribute not in attribute_values]
            #print(not_matches_list)
            for not_match in not_matches_list:
                df[column] = df[column].replace(not_match, -1)

    for column in without_unknown_or_numeric_attributes:

        attribute_values_2 = attributes[attributes['Attribute'] == column]['Value'].tolist()
        dataframe_column_values_2 = df[column].unique().tolist()

        if not all(item in attribute_values_2 for item in dataframe_column_values_2):
            not_matches_list = [attribute for attribute in dataframe_column_values_2 if attribute not in attribute_values_2]
            #print(not_matches_list)
            for not_match in not_matches_list:
                df[column] = df[column].replace(not_match, np.nan)
                
                
    return df


def common_attributes_fc(azdias_df, customers_df, attributes_df):
    """
    This function takes common attributes in azdias, customers and attributes files.
    """
   
    azdias_att = list(azdias_df.columns)
    customers_att = list(customers_df.columns)

    attributes_att  = attributes_df['Attribute'].unique().tolist()

    common_attributes = list(set(attributes_att) & set(azdias_att) & set(customers_att))
        
    return common_attributes


def data_full_adjustment(df):
    """
    Fixing and cleaning Data this is set to any of four datasets
    """
    selected_columns = ''

    with open('common_attributesr.pkl', 'rb') as f:
        common_attributes = pickle.load(f)
        
    with open('selected_columns.pkl', 'rb') as f:
        selected_columns = pickle.load(f)

    with open('attributes.pkl', 'rb') as f:
            attributes = pickle.load(f)
    new_df = df.copy()
    
    new_df = fix_warning_cols(new_df)
    print("Fixed Warning Columns")
    new_df = new_df[common_attributes]
    print("Setting only common attributes: OK!")
    new_df = standardize_dataset(new_df)
    print("Standardize Dataset Values: OK!")
    new_df = feature_engeneering(new_df)
    print("Feature Engeneering: OK!")
    new_df = label_categorical_to_numeric(new_df)
    new_df = replace_unknown_values(new_df)
    
    with open('selected_columns.pkl', 'rb') as f:
        selected_columns = pickle.load(f)
    
    new_df = new_df[selected_columns]
    
    return new_df   


# FEATURE FUNCTIONS

def wealth_feature(df):
    """
    Create 'wealth' feature derived from 'CAMEO_INTL_2015'
    """
    
    wealth_dict = {-1:'unknown',1:'wealthy',2:'prosperous',
                   3:'comfortable',4:'less-affluent',5:'poorer'}
    
    
    df['wealth'] = df['CAMEO_INTL_2015'].apply(lambda x: np.floor_divide(float(x), 10) if float(x) else -1)
    df['wealth'] = df['wealth'].map(wealth_dict)
    
    return df

def family_feature(df):
    """
    Create 'family' feature derived from 'CAMEO_INTL_2015'
    """
    
    family_dict = {-1:'unknown',1:'Pre-Family Couples & Singles',
                   2:'Young Couples With Children',
                   3:'Families With School Age Children',
                   4:'Older Families &  Mature Couples',
                   5:'Elders In Retirement'}
    
    df['family'] = df['CAMEO_INTL_2015'].apply(lambda x: np.mod(float(x), 10) if float(x) else -1)
    #df['family'] = df['family'].replace(np.nan,-1)
    df['family'] = df['family'].map(family_dict)
    df['family'] = df['family'].fillna('unknown')
    
    return df

def neighborhood_feature(df):
    """
    Create 'neighborhood_quality' feature derived from 'WOHNLAGE'
    """
    neigh_dict = {-1: 'unknown', 
                  0: 'unknown', 1:'very_good', 2:'good', 3:'average', 
                  4:'poor', 5: 'very_poor', 7:'rural', 8:'new_rural'}
    
    df['neighborhood_quality'] = df['WOHNLAGE'].map(neigh_dict)
    
    return df

def community_unployement_feature(df):
    """
    Create 'community_unployement' feature derived from 'RELAT_AB'
    """
    
    unployement_dict = {-1: 'unknown', 
                         1:'very_low', 2:'low', 3:'average', 4:'high', 5: 'very_high'}
    
    df['community_unployement'] = df['RELAT_AB'].map(unployement_dict)
    
    return df

def gender_feature(df):
    """
    Create 'gender' feature derived from 'ANREDE_KZ'
    """
    
    
    gender_dict = {-1: 'unknown', 1:'male', 2:'female'}
    
    df['gender'] = df['ANREDE_KZ'].map(gender_dict)

    
    
    return df


def life_stage_feature(df):
    """
    Create 'life_stage' feature derived from 'LP_LEBENSPHASE_FEIN'
    and group it by age
    """
    
    life_dict = {1: 'younger_age', 2: 'middle_age', 3: 'younger_age',
              4: 'middle_age', 5: 'advanced_age', 6: 'retirement_age',
              7: 'advanced_age', 8: 'retirement_age', 9: 'middle_age',
              10: 'middle_age', 11: 'advanced_age', 12: 'retirement_age',
              13: 'advanced_age', 14: 'younger_age', 15: 'advanced_age',
              16: 'advanced_age', 17: 'middle_age', 18: 'younger_age',
              19: 'advanced_age', 20: 'advanced_age', 21: 'middle_age',
              22: 'middle_age', 23: 'middle_age', 24: 'middle_age',
              25: 'middle_age', 26: 'middle_age', 27: 'middle_age',
              28: 'middle_age', 29: 'younger_age', 30: 'younger_age',
              31: 'advanced_age', 32: 'advanced_age', 33: 'younger_age',
              34: 'younger_age', 35: 'younger_age', 36: 'advanced_age',
              37: 'advanced_age', 38: 'retirement_age', 39: 'middle_age',
              40: 'retirement_age'}

    
    df['life_stage'] = df['LP_LEBENSPHASE_FEIN'].map(life_dict)
    
    return df
    
    
    
def income_feature(df):
    """
    Create 'income_category' feature derived from 'LP_LEBENSPHASE_FEIN'
    and group it by income
    """
    income_dict = {1: 'low', 2: 'low', 3: 'average', 4: 'average', 5: 'low', 6: 'low',7: 'average', 8: 'average', 9: 'average', 10: 'wealthy', 11: 'average',
              12: 'average', 13: 'top', 14: 'average', 15: 'low', 16: 'average',17: 'average', 18: 'wealthy', 19: 'wealthy', 20: 'top', 21: 'low',
              22: 'average', 23: 'wealthy', 24: 'low', 25: 'average', 26: 'average', 27: 'average', 28: 'top', 29: 'low', 30: 'average', 31: 'low',
              32: 'average', 33: 'average', 34: 'average', 35: 'top', 36: 'average', 37: 'average', 38: 'average', 39: 'top', 40: 'top'}
    
    df['income_category'] = df['LP_LEBENSPHASE_FEIN'].map(income_dict)
    
    return df


def positioning_feature(df):
    """
    Create 'positioning' feature derived from 'PRAEGENDE_JUGENDJAHRE'
    and group it by something like social positioning
    """
    positioning_dict = {1: 'mainstream',  
                     3: 'mainstream',
                     5: 'mainstream',
                     8: 'mainstream',
                     10: 'mainstream',
                     12: 'mainstream',
                     14: 'mainstream',
                     #################
                     2: 'avantgarde',
                     4: 'avantgarde',
                     6: 'avantgarde', 
                     7: 'avantgarde',
                     9: 'avantgarde',
                     11: 'avantgarde',
                     13: 'avantgarde',
                     15: 'avantgarde',
                     ##################
                     -1: 'unknown'}
    
    df['positioning'] = df['PRAEGENDE_JUGENDJAHRE'].map(positioning_dict)
    
    return df

def feature_engeneering(df):
    """
    Group all features function to transform a dataframe
    """
    
    columns_to_drop = ['PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015','LP_LEBENSPHASE_GROB', 'LP_LEBENSPHASE_FEIN', 'RELAT_AB']
    
    df = wealth_feature(df)
    print("Wealth Feature, ok!")
    
    df = family_feature(df)
    print("Family Feature, ok!")
    
    df = neighborhood_feature(df)
    print("Neighborhood Feature, ok!")
    
    df = community_unployement_feature(df)
    print("Community Feature, ok!")
    
    df = gender_feature(df)
    print("Gender Feature, ok!")
    
    df = life_stage_feature(df)
    print("Life Stage Feature, ok!")
    
    df = income_feature(df)
    print("Income Feature, ok!")
    
    df = positioning_feature(df)
    print("Positioning(Mainstream vs Avantgarde) Feature, ok!")
    
    birthyear_attribute = 'GEBURTSJAHR'
    df[birthyear_attribute] = df[birthyear_attribute].replace(0, np.nan)
    print("Fixing birthdate attribute - replacing zeros to NaNs")
    
    df = df.drop(columns_to_drop, axis=1)
    
    return df


def label_categorical_to_numeric(df):
    """
    Convert: 
    categorical to numeric and 
    unknown variables to NaN
    """
    
    life_stage = {'younger_age': 1, 'middle_age': 2, 'advanced_age': 3,
                'retirement_age': 4, 'unknown':np.nan}

    income_scale = {'low': 1, 'average': 2, 'wealthy': 3, 'top': 4, 'unknown':np.nan}

    position_dict = {'mainstream':1, 'avantgarde':2, 'unknown':np.nan}

    neighborhood_quality = {'very_poor': 1, 'poor': 2, 'average': 3, 'good': 4, 'very_good': 5, 'new_rural': np.nan, 'rural': np.nan, 'unknown':np.nan}
    
    rural = {'very_poor': -1, 'poor': -1, 'average': -1, 'good': -1, 'very_good': -1, 'new_rural': 1, 'rural': 1, 'unknown':np.nan}
    
    community_unemployement = {'very_low': 5, 'low': 4, 'average': 3, 'high': 2, 'very_high': 1, 'unknown':np.nan}

    wealth = {'poorer': 1, 'less-affluent': 2, 'comfortable': 3, 'prosperous': 4, 'wealthy': 5, 'unknown':np.nan}

    OST_WEST = {'O': 1, 'W': 2, 'unknown':np.nan}

    gender = {'female': 1, 'male': 2, 'unknown':np.nan}

    family = {'Elders In Retirement': 5, 
              'Families With School Age Children': 4, 
              'Older Families &  Mature Couples': 3, 
              'Pre-Family Couples & Singles': 2, 
              'Young Couples With Children': 1, 
              'unknown':np.nan}

    cameo = {'1A': 1, '1B': 2, '1C': 3, '1D': 4, '1E': 5, 
             '2A': 6, '2B': 7, '2C': 8, '2D': 9, 
             '3A': 10, '3B': 11, '3C': 12, '3D': 13, 
             '4A': 14, '4B': 15, '4C': 16, '4D': 17, '4E': 18, 
             '5A': 19, '5B': 20, '5C': 21, '5D': 22, '5E': 23, '5F': 24, 
             '6A': 25, '6B': 26, '6C': 27, '6D': 28, '6E': 29, '6F': 30, 
             '7A': 31, '7B': 32, '7C': 33, '7D': 34, '7E': 35, 
             '8A': 36, '8B': 37, '8C': 38, '8D': 39, 
             '9A': 40, '9B': 41, '9C': 42, '9D': 43, '9E': 44}




    df['life_stage'] = df['life_stage'].replace(life_stage).astype(float)
    df['income_category'] = df['income_category'].replace(income_scale).astype(float)
    df['positioning'] = df['positioning'].replace(position_dict).astype(float)
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].replace(OST_WEST).astype(float)
    df['gender'] = df['gender'].replace(gender).astype(float)
    df['wealth'] = df['wealth'].replace(wealth).astype(float)
    df['rural'] = df['neighborhood_quality'].replace(rural).astype(float)
    df['neighborhood_quality'] = df['neighborhood_quality'].replace(neighborhood_quality).astype(float)
    df['family'] = df['family'].replace(family).astype(float)
    df['community_unployement'] = df['community_unployement'].replace(community_unemployement).astype(float)
    df['CAMEO_DEU_2015'] = df['CAMEO_DEU_2015'].replace(cameo).astype(float)
    
    return df



def replace_unknown_values(df):
    """
    Replace all unknown values '-1' to NaN
    """
    df_columns = df.columns

    for column in df_columns:
        df[column] = df[column].replace(-1,np.nan)
        
    return df

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# CLUSTER PLOT FUNCS
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#



def plot_clusters(model, df1,df2,label1='General Population',label2="Customers"):
    """
    This function plot the population clusters in datasets
    :params:
    model: KMeans model. Fitted model for clusters
    df1, df2: Pandas DataFrame. Dataframes to compare and plot
    label1, label2: String. Strings to show in the plot
    :return:
    None
    """

    azdias_K = list(model.predict(df1))
    for i in range(len(azdias_K)):
        azdias_K[i] += 1
    customers_K =  list(model.predict(df2))
    for i in range(len(customers_K)):
        customers_K[i] += 1

    data_label = []
    for i in range(len(df1)):
        data_label.append('General Population')
    for i in range(len(df2)):
        data_label.append('Customers')

    data_clusters = azdias_K + customers_K

    data_dict = {'Data':data_label,
                 'Cluster':data_clusters}

    df_clusters = pd.DataFrame(data_dict)
    pop_data1 = list(df_clusters[df_clusters['Data']=='General Population']['Data'].values)
    customers_data1 = list(df_clusters[df_clusters['Data']=='General Population']['Cluster'].values)
    pop_data2 = list(df_clusters[df_clusters['Data']=='Customers']['Data'].values)
    customers_data2 = list(df_clusters[df_clusters['Data']=='Customers']['Cluster'].values)


    pop_dict = dict(Counter(azdias_K))
    c_dict = dict(Counter(customers_K))
    tot_pop = len(azdias_K)
    tot_customers = len(customers_K)

    azdias_k_perc = []
    customers_k_perc = []

    for i in list(set(customers_K)):
        item = pop_dict[i]
        item = round(item / tot_pop, 2)
        azdias_k_perc.append(item)

    for i in list(set(customers_K)):
        item = c_dict[i]
        item = round(item / tot_customers, 2)
        customers_k_perc.append(item)

    plt.rcParams['figure.figsize'] = [15, 7]

    labels = list(set(customers_K))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, azdias_k_perc, width, label=label1)
    rects2 = ax.bar(x + width/2, customers_k_perc, width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Clusters')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    fig.save("Image_1.png")
    fig.tight_layout()
    
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# PCA FUNCTIONS                                                                                      #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#



def pca_model(df, n_components):
    '''
    This function defines a model that takes in a dataframe and returns model fitted in a dataframe. 
    :param:
    df: Pandas DataFrame. Dataframe to be fitted
    n_components: Integer. Number of components to fit
    :return:
    pca: KMeans(). Fitted model
    '''
    pca = PCA(n_components, svd_solver='full')
    
    return pca.fit(df)

def scree_plots(df,dataname):
    '''
    This function takes in the transformed data using PCA and plots it in scree plots
    :params:
    df: PandasDataframe. PCA transformed data
    dataname: String. String used in the tittle
    :return None:
    
    '''
    plt.rcParams["figure.figsize"] = (20,20)
    subplot(2,1,1)

    plt.plot(np.cumsum(df.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components ' + dataname)
    plt.grid(b=True)


    plot = tight_layout()
    plot = plt.show()
    savefig('PCA_feat.png',format='png',dpi=200)
    
def display_interesting_features(df, pca, dimensions,show=5):
    '''
    This function displays interesting features of the selected dimension
    :params:
    df: PandasDataFrame. Base Data
    pca: PCA model
    dimensions: Integer. Number of dimensions
    show: Integer. Number of features to show
    '''
    
    features = df.columns
    components = pca.components_
    feature_weights = dict(zip(features, components[dimensions]))
    sorted_weights = sorted(feature_weights.items(), key = lambda kv: kv[1])
    
    print('Lowest: ')
    for feature, weight, in sorted_weights[:show]:
        print('\t{:20} {:.3f}'.format(feature, weight))
    
    print('Highest: ')
    for feature, weight in sorted_weights[show*-1:]:
        print('\t{:20} {:.3f}'.format(feature, weight))
        
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#        
# KMEANS FUNC
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#



def fill_nan_with_mean(df):
    """
    Fill NaN values with the mean value of the feature
    :param:
    df: PandasDataFrame. Data to be filled
    :return:
    df: PandasDataFrame. dataframe without NaN values
    """
    for column in df.columns:
        df[column].fillna(float(df[column].mean()),inplace=True)
    
    return df

def fill_nans_kmeans(df):
    """
    Fill NaN values with the mean value of the feature in the cluster
    """
    
    with open('kmodel_fill_clusters.pkl', 'rb') as f:
        kmeans = pickle.load(f)
        
    with open('azdias_scaler.pkl', 'rb') as f:
        azdias_scaler = pickle.load(f)
        
    
    new_df = df.copy()
    new_df = fill_nan_with_mean(new_df)
    new_df = pd.DataFrame(azdias_scaler.transform(new_df))
    new_df.columns = selected_columns
    
    valid = np.isfinite(new_df.copy().values)

    azdias_K = list(kmeans.predict(new_df))
    
    new_df = np.where(valid, new_df.values, np.nan)
    new_df = pd.DataFrame(new_df)
    new_df.columns = selected_columns
    
    new_df['Cluster'] = azdias_K
    clusters = list(new_df['Cluster'].sort_values().unique())
    
    for cluster in clusters:
        for column in selected_columns:
            index = new_df.loc[new_df['Cluster']==cluster][column]
            index = index[index.isnull()].index
            mean_value = new_df[new_df['Cluster']==cluster][column].mean()
            new_df.loc[index,column] = mean_value

        print(f"Cluster nº{cluster} - OK")
    
    new_df = new_df.drop(['Cluster'], axis=1)
    
    return new_df