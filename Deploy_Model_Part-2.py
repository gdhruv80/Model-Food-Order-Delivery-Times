import json
import ast
import os
import sys
import pandas as pd
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Supportin Functions used ###
def ds_summry(Mod_df_2):
    msng = ((Mod_df_2.isnull().sum()/len(Mod_df_2))*100).apply(round,args=(2,))
    uniq,max_v,min_v = Mod_df_2.nunique(),Mod_df_2.max(),Mod_df_2.min()
    df_stats = pd.concat([msng, uniq,max_v,min_v], axis=1).rename(index=str, 
                                                                  columns={0: "% missing", 1: "No_uniq",2:'Max_Val',3:'Min_Val'})
    return df_stats
def shift(x):
    if x >6 and x<12:
        return 1    # 'Morning'
    elif x >=12 and x<=17:
        return 2    #'Lunch'
    elif x >17 and x<24:
        return 3    #'Dinner'
    else:
        return 4    #'Early_Morning'

# Main Scoring functions ###

#####   STAGE 1 - LOADING THE JSON DATA #####
def stg1(json_dat):
    
    inp_dat = []
    with open(json_dat) as json_file: 
        for i in json_file:
            inp_dat.append(ast.literal_eval(i))

    # Creating a df of the scoring data
    scr_df = pd.DataFrame(inp_dat)

    for i in scr_df.columns:
        scr_df.loc[scr_df[i] == 'NA',i] = float('nan')
    return scr_df


#####   STAGE 2 - CLEANING/IMPUTING THE INPUT DATA #####
def stg2(scr_df):
    #Correcting for negetive values 
    for i in ['min_item_price','total_outstanding_orders','total_busy_dashers','total_onshift_dashers']:
        scr_df[i] = scr_df[i].apply(float)
        scr_df[i] = scr_df[i].apply(lambda x:x if x>=0 else None)

    # Checking for missing values ###
    # print scr_df.isnull().sum()
    # Replace missing values with medians or 0(for categorical variables - additional ctgry created) from the model creation file

    # Replacing with 0(creating a new category) like in the modelling file
    scr_df.loc[scr_df['market_id'].isnull(),'market_id'] = 0
    scr_df.loc[scr_df['order_protocol'].isnull(),'order_protocol'] = 0
    scr_df.loc[scr_df['store_primary_category'].isnull(),'store_primary_category'] = 'miss'

    # Replacing with median
    scr_df.loc[scr_df['total_items'].isnull(),'total_items'] = 3     
    scr_df.loc[scr_df['subtotal'].isnull(),'subtotal'] = 2200
    scr_df.loc[scr_df['num_distinct_items'].isnull(),'num_distinct_items'] = 2
    scr_df.loc[scr_df['min_item_price'].isnull(),'min_item_price'] = 595
    scr_df.loc[scr_df['total_onshift_dashers'].isnull(),'total_onshift_dashers'] = 37
    scr_df.loc[scr_df['total_busy_dashers'].isnull(),'total_busy_dashers'] = 34      
    scr_df.loc[scr_df['max_item_price'].isnull(),'max_item_price'] = 1095
    scr_df.loc[scr_df['total_outstanding_orders'].isnull(),'total_outstanding_orders'] = 41
    scr_df.loc[scr_df['estimated_order_place_duration'].isnull(),'estimated_order_place_duration'] = 251
    scr_df.loc[scr_df['estimated_store_to_consumer_driving_duration'].isnull(),'estimated_store_to_consumer_driving_duration'] = 544

    scr_df['total_onshift_dashers'] = np.where(scr_df['total_busy_dashers']>scr_df['total_onshift_dashers'],
                                                 scr_df['total_busy_dashers'],scr_df['total_onshift_dashers'])

    return scr_df


#####   STAGE 3 - ENCODING THE DISCRETE VARIABLES AND FEATURE ENGINEERING #####
def stg3(scr_df):
    
    #labeling for store_primary_category and store_id
    store_id_mod = pickle.load(open("Label_enc_storeid.sav",'rb'))
    store_prm_ctg_mod = pickle.load(open("Label_store_prm_ctgy.sav",'rb'))

    scr_df['store_id'] = scr_df['store_id'].apply(lambda x: store_id_mod.transform([x])[0] if x in store_id_mod.classes_ else float('nan'))

    scr_df.loc[np.isnan(scr_df['store_id']), 'store_id'] = scr_df['store_id'].median(skipna = True)

    scr_df['store_primary_category'] = scr_df['store_primary_category'].apply(lambda x: x if x in store_prm_ctg_mod.classes_ 
                                                                              else 'miss')
    scr_df['store_primary_category'] = store_prm_ctg_mod.transform(scr_df['store_primary_category'])

    # Creating synthetics/feature engineering

    scr_df['created_at'] = pd.to_datetime(scr_df['created_at'])
    scr_df['Deliv_Day'] = scr_df['created_at'].dt.weekday
    scr_df['Deliv_hour'] = scr_df['created_at'].dt.hour

    scr_df['Deliver_shift'] = scr_df['Deliv_hour'].apply(shift)

    # Creating synthetics to check how busy the delivery service or dashers are at any given time

    scr_df['%_Dashers_free'] = np.where((scr_df['total_busy_dashers'] + scr_df['total_onshift_dashers'] ==0) , 0,
                                          (1 - (scr_df['total_busy_dashers']/scr_df['total_onshift_dashers']))*100)

    scr_df['Free_Dash/Outdng_Orders'] = np.where((scr_df['total_outstanding_orders'] ==0) , 0,
                                                   (scr_df['total_onshift_dashers'] - 
                                                    scr_df['total_busy_dashers'])/scr_df['total_outstanding_orders'])
    
    return scr_df


#####   STAGE 4 - SCALING AND RUNNING/SCORING THE MODEL #####

def stg4(scr_df):
    # Selecting the modelling subset

    cols =
    ['market_id','order_protocol','total_items','subtotal','num_distinct_items','min_item_price','max_item_price','total_onshift_dashers',
     'total_busy_dashers','total_outstanding_orders','estimated_order_place_duration','estimated_store_to_consumer_driving_duration',
     'store_id','store_primary_category','Deliv_Day','Deliv_hour','Deliver_shift','%_Dashers_free','Free_Dash/Outdng_Orders']

    # scalling the cols
    X = scr_df[cols]
    scal_coef = pickle.load(open("Scalling_coef.sav",'rb'))
    X_scal = scal_coef.transform(X)

    # Running the model
    model = pickle.load(open("Doordash_Final_model_rf.sav",'rb'))
    output = model.predict(X_scal)
    # outputting the results as tsv
    out_df = pd.concat([scr_df['delivery_id'],pd.Series(output)],axis = 1)
    out_df.rename(index=str, columns={0: "Predicted_Time_secs"},inplace = True)
    out_df.to_csv('Scored_Output_n.csv',sep = ' ',index = False)

    return 'The model has sucessfully scored and the tsv file generated in the same folder as this py file'

def score(json_dat):
    s1 = stg1(json_dat)
    s2 = stg2(s1)
    s3 = stg3(s2)
    return stg4(s3)

if __name__ == '__main__':
    score(sys.argv[1])