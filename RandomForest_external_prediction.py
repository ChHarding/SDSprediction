# Batch run making models and predicting them
# Chris Harding,  Nov 2019
# Code after Mohsin's Defense, working on full paper

import sys, os
import time
import numpy as NUM
import numpy as np
import pandas as PD
import pandas as pd
import matplotlib.pyplot as PLOT
import matplotlib.pyplot as plt
import seaborn as SEA
import seaborn as sns
import eli5
import arcpy  

import warnings # suppress depreciation warnings, need to be don BEFORE import!
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn.ensemble import RandomForestClassifier # module to install is called scikit-learn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz


import pydotplus
from io import StringIO

from os.path import abspath
from collections import OrderedDict 
from collections import Counter 

def import_featureclass_to_dataframe(feature_class, workspace, column_names, new_names, 
                                     dummify=None, add_NDVI=True):
    """ reads in a featureclass in a workspace and returns a dataframe (with updated variable names)
    column_names: list of columns that shall be read in from the featureclass
    new_names: change dict for changing variable names
    add_NDVI: flag calculating a new NDVI column
    dummify: string with name of int variable to be converted to a set of dummy (binary) vars
    """
 

    arcpy.env.workspace = workspace
    fc_np = arcpy.da.FeatureClassToNumPyArray(feature_class, column_names)

    # Convert numpy array to Pandas dataframe
    data = PD.DataFrame(fc_np, columns=column_names)

    # use better names for the band numbers (NIR = Near Infrared)
    data.rename(columns=new_names, inplace=True)

    if add_NDVI:
        # Calculate NDVI and put it in a new column
        ndvi = (data["NIR"] - data["Red"]) / (data["NIR"] + data["Red"])
        data.insert(5, 'NDVI', ndvi)

    if dummify == None:
        # Rotation is a categorical variable with 3 different levels that encodes the type of crop rotation
        # used in each quadrant. It is initally a string but it is easier if each level is encoded as an integer
        def tran_Rotation(x):
            if x == 'S2':
                return 2
            if x == 'S3':
                return 3
            if x == 'S4':
                return 4

        data['Rotation'] = data['Rotation'].apply(tran_Rotation)
        data['Rotation'] = data['Rotation'].astype('category')
    else:
        assert dummify in list(data), "Error: " + dummify + "not in" + str(list(data))

        data['Rotation'] = data['Rotation'].astype('category')

        # As sklearn's random forest models cannot deal with categorical variables, we create
        # 3 dummy (binary) columns instead
        rotation = data[dummify] # save rotation column
        data = data.drop(dummify, axis="columns") # then delete it


        # Make dummy (binary) dataframe with colums for S2, S3 and S4
        num_rows = len(data.index)
        rot_types = sorted(list(rotation.unique())) # ["S2", "S3", "S4"] # types of crop rotation to encode
        dummy = PD.DataFrame(0, index = np.arange(num_rows), # init cols with 0 
                            columns = rot_types)

        # set 1 for each of the columns if they correspond to the value in rotation
        for t in rot_types: #columns
            for ri in range(0, num_rows): # row index
                if rotation.at[ri] == t:  # e.g. "S2"
                    dummy.at[ri, t] = 1 # set column S2 to 1 
        #printLog(rotation.head())
        #printLog(dummy.head())

        # insert dummy columns just to the left of the last column
        num_cols = len(list(data))
        for c in rot_types[::-1]:
            data.insert(loc=num_cols-1, column=c, value=dummy[c])
            
  
    
    return data 

# using the decision tree t and a single(!) quadrat q, predict the response variable  (SDS) from the same
# set of exploratory variables used to create the rf model the tree is part of.
# returns 1 (SDS) or 0 (healthy)
def predict(t, q):
    a = np.array(q)# numpy 1D array
    a = a.reshape(1, -1)# turn into 2D array otherwise I get ValueError: Expected 2D array, got 1D array instead:
    r = t.predict(a) # float array with 1 element
    p = t.predict_proba(a) #???
    return int(r[0])



def list_feature_classes(workspace):
    # list all polygon layers in the workspace
    printLog("workspace contains these polygon layers:")
    walk = arcpy.da.Walk(workspace, datatype="FeatureClass", 
                        type="Polygon")

    for dirpath, dirnames, filenames in walk:
        if filenames != []:
            printLog(os.path.basename(dirpath))
            for f in filenames:
                printLog("\t",f)
            printLog()

# functions 
logfilehandle = None
def printLog(*args, **kwargs):
    """ print to console and into file if global var logfilehandle is not None
    logfilehandle needs to be opened and closed externally! e.g.
    logfilehandle = open('output.out','a+')
    """
    print(*args, **kwargs)
    if logfilehandle:
        print(*args, **kwargs, file=logfilehandle)

def change_expl_var_names(expl_vars, change_name_dict):
    """changes expl_vars according to a change-dict and returns list of names
    expl_vars:  ['MEAN_1', 'MEAN_2', 'MEAN_3', 'Rotation']
    new_names = {'MEAN_1': 'Blue', 'MEAN_2': 'Green', 'MEAN_3': 'Red', 'MEAN_4': 'NIR'}
    returns: ['Blue', 'Green', 'Red', 'Rotation']
    """
    changed_names = []
    for var in expl_vars:
        # check if var needs to be changed
        has_changed = False
        for k in change_name_dict.keys():
            if var == k: # var name needs to be changed
                changed_names.append(change_name_dict[k]) # push changed name and 
                has_changed = True
                break
        if not has_changed: changed_names.append(var) # push old name, no change was done
    return changed_names

def graph_decision_trees(pred_dict, dir="."):
    """ creates a list of graphiviz graph objects, one for each tree in rf
    pred_dict: prediction to graph
    dir: folder to write images into
    name of tree img will be prediction date + tree (01, 02, etc.)
    """
    p = pred_dict 
    pd_str = p["pred_date"][0]
    mn = p["model_name"]
    rf = p["tuned_model"]
    var_names=p["expl_vars"]
    for i in range(rf.n_estimators):
    
        # Extract the  tree
        estimator = rf.estimators_[i]
        #printLog(estimator)
        
        # Create a buffer with a .dot file 
        dot_data_str = export_graphviz(estimator,
                        out_file=None,
                        feature_names=var_names, # 
                        class_names = ['healthy', 'diseased'],
                        rounded=True,
                        proportion=True,
                        precision=2,
                        filled=True,
                        special_characters=True)

        dotgraph = pydotplus.graph_from_dot_data(dot_data_str)
    
        # save the graph as a png file
        filename = dir + os.sep + pd_str + "_from_" + mn + str("_%02d" % i) + ".png"
        dotgraph.write_png(filename) # save as png file to disk
    
    

# make a tuned model for this prediction
def tune_model(prediction_dict, data_dict):   
    """ using data from data_dict, creates an tuned model for the model dates
    in prediction_dict. 
    The tuned model is returned but also stored in "tuned_model"
    report dict "rep" is filled with internal prediction quality metrics
    """  
    p = prediction_dict     
    mdl = p["model_date_list"][:] # copy(!) of model dates 
    if mdl == []: return None

    pd_str = p["pred_date"][0]
    # remove pred date from list of model dates?
    if p["no_same_date"] == True:
        dates_only_list = [x[0] for x in mdl]
        i = dates_only_list.index(pd_str) if pd_str in dates_only_list else None
        if i != None: #found a same date
            printLog("removed date", mdl[i][0])
            del mdl[i] # delete that date

    model_dates_s = ""
    for m in mdl:
        md_str = m[0]
        model_dates_s += md_str + " "
    printLog("Model dates:", model_dates_s)
    
    # fill report ordered dict
    rep = p["report"]
    rep["model_dates"] = model_dates_s
    rep["model_dates_l"] = p["model_date_list"]

    # if needed glue together multi-date model data
    df_list = []
    for m in mdl:
        md_str = m[0]  # model date string
        df_list.append(data_dict[md_str])
    model_data = PD.concat(df_list, sort=False)
    prediction_dict["model_data"] = model_data

    pred_data = data_dict[pd_str]
    prediction_dict["pre_data"] = pred_data

    expl_vars = p["expl_vars"]
    resp_var = p["resp_var"]
    printLog("predicting", resp_var, "from", expl_vars)
    rep["expl_vars"] = " ".join(p["expl_vars"])
    rep["expl_vars_l"] = p["expl_vars"]
    rep["resp_var"] = p["resp_var"]

    # made dataframes from these columns
    X = model_data[expl_vars] # Explanatory variables
    y = model_data[resp_var]  # Response variable

    # Split dataset into training set and test set
    SPLIT_RND_SEED = 12345
    test_size = p["test_size"] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=SPLIT_RND_SEED) 
    printLog("Using", len(X_train), "quadrants for training,", len(y_test), "quadrants for testing")
    rep["n_total"] = len(X_train) + len(y_test)
    rep["n_training"] = len(X_train)
    rep["n_test"] = len(y_test)

    # create model
    model = RandomForestClassifier( n_jobs=-1, 
                                    random_state=12345, 
                                    bootstrap=False, # use full dataset. Default True
                                    verbose=0) # needs verbose=0 or gridsearch will by very chatty!

    # optimizer
    printLog("grid search parameter list", p["grid_params"])
    rf_gridsearch = GridSearchCV(estimator=model, 
                                    param_grid=p["grid_params"],
                                    scoring='roc_auc',
                                    n_jobs=-1,
                                    cv=5,  # folds
                                    verbose=1, 
                                    return_train_score=True)
    rf_gridsearch.fit(X_train, y_train)
    best_params = rf_gridsearch.best_params_
    printLog("best parameters", best_params)

    # put param values in report
    grid_params2 = p["grid_params"].copy() # make a copy, just to be safe ...
    del grid_params2['class_weight'] # remove class_weight
    for bp in grid_params2: 
        rep[bp] = "%d" % best_params[bp]


    # create optimized model from optimized params
    rf = RandomForestClassifier(**best_params, 
                            oob_score=True, 
                            random_state=12345, 
                            verbose=0)
    p["tuned_model"] = rf

    # Train the tuned model using the training sets
    c = rf.fit(X_train, y_train)
    printLog("Tuned model:", c)
    rep["tuned_model"] = str(rf)
    

    #       
    # Evaluate tuned model 
    #
    printLog()

    # Accuracy  (ratio of correct predictions made using the model)
    acc_test = rf.score(X_test, y_test) # test subset
    printLog('Accuracy on the test subset:', acc_test)
    rep["acc_test"] = acc_test

    acc_train = rf.score(X_train, y_train) # training subset
    printLog('Accuracy on the training subset:', acc_train)
    rep["acc_train"] = acc_train

    y_pred_all = rf.predict(X) # predict the entire dataset
    acc_full = accuracy_score(y, y_pred_all)
    printLog('Accuracy on the entire data', acc_full)
    rep["acc_full"] = acc_full
    # again but now use probability for SDS
    y_pred_prob = rf.predict_proba(X) # ex: [0.6, 0.4], 0.6 prob for 0, 0.4 for 1
    sds = (y_pred_prob[:,1] > 0.5).astype(int)
    #printLog(accuracy_score(y, sds))
    

    #oob accuracy
    printLog('Out-of-bag score estimate:', rf.oob_score_)
    rep["acc_oob"] = rf.oob_score_


    # confusion matrix using predictins from test only
    y_pred = rf.predict(X_test)
    cm = PD.DataFrame(confusion_matrix(y_test, y_pred))
    printLog("\nConfusion (error) matrix of prediction:\n", cm)
    rep["TN"] = cm[0][0] # true neg
    rep["FN"] = cm[1][0] # false neg
    rep["TP"] = cm[1][1] # true pos
    rep["FP"] = cm[0][1] # false pos

    #  statistics of precision, specificity and sensitivity
    printLog(classification_report(y_test, y_pred,
                                labels=None,
                                target_names=["Healthy", "SDS"],
                                sample_weight=None,
                                digits=2,
                                output_dict=False))
    stats = classification_report(y_test, y_pred,
                                labels=None,
                                target_names=["Healthy", "SDS"],
                                sample_weight=None,
                                digits=2,
                                output_dict=True) # also need dict version
    sensitivity = stats["Healthy"]["recall"]
    specificity = stats["SDS"]["recall"]
    printLog("specificity (correct 1 (SDS) predictions)", specificity, 
          "\nsensitivity (correct 0 (Healthy) predictions)", sensitivity)
    rep["specificity"] = specificity
    rep["sensitivity"] = sensitivity

    # Receiver Operator Characteristics
    probs = rf.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    printLog('\nReceiver Operator Characteristic (ROC) AUC score: ', auc)
    rep["auc"] = auc

    # Kappa
    cohen_score = cohen_kappa_score(y_test, y_pred)
    printLog("\nKappa score:", cohen_score)
    rep["kappa"] = cohen_score

    # Variable importance
    vi = PD.DataFrame({'variable name': list(X_test.columns),
                    'importance': rf.feature_importances_})
    vi = vi.sort_values('importance', ascending = False)
    vi = vi.reset_index(drop=True) 
    printLog(vi)
    rep["var_importance"] = " ".join(["%s:%.3f" % (vi["variable name"][r],vi["importance"][r]) for r in [0,1,2]])

    return rf

def make_data_dict(prediction_list):
    """ return a dict containing all dataframes used in model(s) and predictition(s)
        dict key is the date string of the model or prediction
    """
    data_dict = {}
    for p in prediction_list:
        # columns to read in, pre-prend Quadrat which is needed for ArgGIS joins later
        column_names = ["Quadrat"] + p["expl_vars"] + [ p["resp_var"] ]
        add_NDVI = p["add_NDVI"]
        dummify_var = "Rotation"

        # read in prediction data
        pd_str = p["pred_date"][0]  # pre date as str
        pd_fc = p["pred_date"][1] # feature class for that pred date\
        if not pd_str in data_dict: # add data for prediction date if we don't have it yet
            #new_names = {'MEAN_1': 'Blue', 'MEAN_2': 'Green', 'MEAN_3': 'Red', 'MEAN_4': 'NIR'}
            new_names = {}
            d = import_featureclass_to_dataframe(pd_fc, workspace, column_names, new_names, 
                                                 dummify_var, add_NDVI)
            for n in ["Quadrat", "SDS"]: d[n] = d[n].astype("int")
            data_dict[pd_str] = d#.sample(n=21)

        # read in model data
        mdl = p["model_date_list"]
        for m in mdl:
            md_str = m[0]  # model date string
            md_fc = m[1] # feature class for that model date
            if md_str in data_dict: continue  # skip if we have that date already
            d = import_featureclass_to_dataframe(pd_fc, workspace, column_names, new_names, 
                                                 dummify_var, add_NDVI)
            for n in ["Quadrat", "SDS"]: d[n] = d[n].astype("int")
            data_dict[md_str] = d#.sample(n=21)
        
        # Update to changed (new) names of expl vars
        p["expl_vars"] = list(data_dict[pd_str]) 
        for n in ["Quadrat", "SDS"]: # remove non explanatory variables!
            p["expl_vars"].remove(n)


    return data_dict

# predict external data from turned model
def external_prediction(prediction_dict, data_dict):
    p = prediction_dict     
    mdl = p["model_date_list"]
    if mdl == []: return None

    pd_str = p["pred_date"][0]
    rep = p["report"]
    printLog("\nExternal prediction of", pd_str)
    rep["pred_date"] = pd_str
    
    vdata = data_dict[pd_str] # fetch data for validation
    rep["ext_num_samples"] = len(vdata.index)
    printLog("External prediction samples", rep["ext_num_samples"]) 
    expl_vars = p["expl_vars"]
    resp_var = p["resp_var"]
    printLog("predicting", resp_var, "from", expl_vars)
    
    X = vdata[expl_vars] # Explanatory variables df
    y = vdata[resp_var]  # Response variable df
    rep["ext_n_total"] = len(X.index)

    # predict external prediction data using the tuned model
    rf = p["tuned_model"]
    y_pred = rf.predict(X) 

    acc  = accuracy_score(y, y_pred)
    printLog('External accuracy', acc)
    rep["ext_acc"] = acc

    # Use prediction probabilities
    y_pred_prob = rf.predict_proba(X) # [0.6, 0.4] 0.6 prob for 0, 0.4 for 1
    prob_healthy_min = y_pred_prob[:,0].min()
    prob_healthy_mean= y_pred_prob[:,0].mean()
    prob_healthy_max = y_pred_prob[:,0].max()
    prob_sds_min = y_pred_prob[:,1].min()
    prob_sds_mean= y_pred_prob[:,1].mean()
    prob_sds_max = y_pred_prob[:,1].max()
    rep["ext_healthy_prob"] = "%.3f %.3f %.3f" % (prob_healthy_min, prob_healthy_mean, prob_healthy_max)
    rep["ext_sds_prob"] = "%.3f %.3f %.3f" % (prob_sds_min, prob_sds_mean, prob_sds_max)
   
    # same as rf.predict(X) but could instead use 0.6 etc.
    y_pred = (y_pred_prob[:,1] > 0.5).astype(int) # cast from True/False to 1/0
    acc  = accuracy_score(y, y_pred)
    rep["ext_acc2"] = acc

    # extend vdata with columns for results
    newdf = PD.DataFrame(columns = ["ext_SDS", # prediction of SDS 0 or 1 
                                    "ext_prb", # probablity of prediction
                                    "ext_cond",# condition: TP, FP, TN, FN
                                   ])
    vdata =  PD.concat([vdata, newdf], sort=False)

    vdata["ext_SDS"] = y_pred # store SDS prediction
    vdata["ext_prb"] = y_pred_prob

    # classifiy each predicted quadrat as TP, TN, FP or FN
    for i in vdata.index:  # i is row
        truth = vdata.at[i,"SDS"]
        pred = vdata.at[i,"ext_SDS"]

        if truth == pred:
            if truth == 1:
                vdata.at[i,"ext_cond"] = "TP" # true positive
            else:
                vdata.at[i,"ext_cond"] = "TN" # true negative
        else:
            if truth == 1:
                vdata.at[i,"ext_cond"] = "FN" # true positive
            else:
                vdata.at[i,"ext_cond"] = "FP" # true negative

    cnd = vdata["ext_cond"]  

    cond_counter = Counter(cnd) # Counter({'TP': 133, 'TN': 88, 'FN': 15, 'FP': 4})
    for cond in cond_counter:
        rep["ext_" + cond] = cond_counter[cond]

    TP = cond_counter["TP"]
    FP = cond_counter["FP"]
    TN = cond_counter["TN"]
    FN = cond_counter["FN"]

    printLog("conditions: TP %d  FP %d  TN %d  FN %d" % (TP, FP, TN, FN))
    rep["ext_TP"] = TP
    rep["ext_FP"] = FP
    rep["ext_TN"] = TN
    rep["ext_FN"] = FN

    rep["ext_specificity"] = TP / (TP + FN)
    rep["ext_sensitivity"] = TN / (TN + FP)

    # Informedness https://en.wikipedia.org/wiki/Youden%27s_J_statistic
    rep["ext_YourdansJ"] = rep["ext_specificity"] + rep["ext_sensitivity"] -1

    printLog("specificity:", rep["ext_specificity"])
    printLog("sensitivity:", rep["ext_sensitivity"])
    
    printLog("Yourdan's J:", rep["ext_YourdansJ"])
    return vdata

def results_as_csv(p, columns):
    """ columns are keys for dict d
    returns string with column values as csv"""

    s = ""
    r = p["report"]
    for c in columns:
        #printLog(c,r[c])
        col_text = str(r[c])
        s += col_text + ", "
    return s[:-2]

####################################

# MAIN

#workspace = r"..\SDS_detection_ArcGISPro_project\SDS_detection_ArcGISPro_project.gdb" # must contain all featureclasses
workspace = r"SDS_detection_ArcGISPro_project.gdb"

# Names of explanatory variables as shown in the fc attribute table
#expl_vars = ['MEAN_1', 'MEAN_2', 'MEAN_3', 'MEAN_4', 'Rotation'] 
expl_vars = ['Rotation', 'x', 'y'] 
for n in range(1,5):
    for v in ["MEAN", "STD", "MIN", "MAX", "MEDIAN"]:
        expl_vars.append(v + "_" + str(n))
resp_vars = ['SDS'] # name of response Variable

join_field = "Quadrat" # key for joining the prediction results back to the prediction fc
add_NDVI = False  # calculate NDVI for each date?
no_same_date = False # set to True to remove dates from models that are also predicted


dates = [
    ["20160720", "Soybean_Quadrats_2016_zstats_cr_T20160720_161306_0e0e_3B_AnalyticMS_SR"],
    ["20160821", "Soybean_Quadrats_2016_zstats_cr_T20160821_161512_0e26_3B_AnalyticMS_SR"],
    ["20160831", "Soybean_Quadrats_2016_zstats_cr_T20160831_161520_0e20_3B_AnalyticMS_SR"],

    ["20170720", "Soybean_Quadrats_2017_zstats_cr_T20170720_161916_101e_3B_AnalyticMS_SR"],

    ["20180716", "Soybean_Quadrats_2018_zstats_cr_T20180716_163325_101b_3B_AnalyticMS_SR"], 
    ["20180724", "Soybean_Quadrats_2018_zstats_cr_T20180724_163356_1004_3B_AnalyticMS_SR"], 
    ["20180822", "Soybean_Quadrats_2018_zstats_cr_T20180822_161812_0f46_3B_AnalyticMS_SR"], 
    ["20180829", "Soybean_Quadrats_2018_zstats_cr_T20180829_163535_1021_3B_AnalyticMS_SR"], 
    ["20180831", "Soybean_Quadrats_2018_zstats_cr_T20180831_163515_1006_3B_AnalyticMS_SR"],  
]


#
# Create a list of prediction dicts
#

# Each dict contains: 
#   a list of on or more dates for the model
#   name of the model (same as date for only 1 date but a summary name for multi-date models)
#   model dataframe
#   prediction date
#   prediction data frame
#   no_same_date flag (if True, the pred date is removed from model dates, if no date is left, not prediction is done)
#   list of exploratory variables
#   add_NDVI ?
#   response variable
#   parameters for grid search optimization
#   report dictionary (ordered)


# Configuration 1: predict all dates from all others dates using single date models
prediction_list = [] # list of all predictions to be worked on
for d1 in dates:
    for d2 in dates:
        repD = OrderedDict() 
        predD = {   
            "model_date_list": [d1], # must be a list even with only 1 date!
            "model_name": d1[0],  # date string
            "test_size": 0.3, # ratio of test vs training subsets
            "model_data": None, 
            "pred_date": d2,
            "pred_data": None,
            "no_same_date": False,
            "expl_vars":  expl_vars, #['MEAN_1', 'MEAN_2', 'MEAN_3', 'MEAN_4', 'Rotation'] ,
            "add_NDVI": False,
            "resp_var": "SDS",
            "tuned_model": None,
            "grid_params": {'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                            'max_features': [1,3,6], # rule of thumb: sq.root of num of vars, small reduces overfitting
                            #'max_depth': [3, 5, 7, 9, 11, 13, 15], # None, split until min_samples_split is reached
                            'min_samples_split':[0.025, 0.05, 0.075], # samples needed for split internal node, in %
                            'min_samples_leaf': [0.025, 0.05, 0.075], # samples at leaf, in %
                            'class_weight': ['balanced'],
                            },

            #"grid_params": {'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            #                'max_features': [1, 3, 6, 9, 12, 15, 18], # rule of thumb: sq.root of num vars, small reduces overfitting
            #                'max_depth': [3, 5, 7, 9, 11, 13, 15], # None, split until min_samples_split is reached
            #                'min_samples_leaf': [1,3,5],
            #                'class_weight': ['balanced'],
            #               },
                            # min_samples_split

#            "grid_params": {    'n_estimators': [10, 40, 70],
#                                'max_features': [2, 4],
#                                'max_depth': [5, 7, 10],
#                                'min_samples_leaf': [2, 5, 10],
#                                'class_weight': ['balanced'], # 
#                            },
            "report": repD , 
        }
        prediction_list.append(predD)

add_NDVI = False  # calculate NDVI for each date?
no_same_date = False # set to True to remove dates from models that are also predicted
repname = "1_from_1_25vars_v2"
'''
# Configuration 2: predict all dates from a multi-date model
prediction_list = [] # list of all predictions to be worked on

model_date_list = dates#[0:3] # 2016
pred_date_list = dates#[3:] # 2017, 2018
add_NDVI = False  # calculate NDVI for each date?
no_same_date = True # set to True to remove dates from models that are also predicted

for pd in pred_date_list:
    repD = OrderedDict() 
    predD = {   
        "model_date_list": model_date_list,
        "model_name": "all",  # date string
        "test_size": 0.3, # ratio of test vs training subsets
        "model_data": None, 
        "pred_date": pd,
        "pred_data": None,
        "no_same_date": no_same_date,
        "expl_vars":  expl_vars, #['MEAN_1', 'MEAN_2', 'MEAN_3', 'MEAN_4', 'Rotation'] ,
        "add_NDVI": False,
        "resp_var": "SDS",
        "tuned_model": None,

#        "grid_params": {    'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#                            'max_features': [1, 3, 6, 9, 12, 15, 18],
#                            'max_depth': [3, 5, 7, 9, 11, 13, 15],
#                            'min_samples_leaf': [1,3,5],
#                            'class_weight': ['balanced'], # 
#                        },
            "grid_params": {    'n_estimators': [10, 40, 70],
                                'max_features': [2, 4],
                                'max_depth': [5, 7, 10],
                                'min_samples_leaf': [2, 5, 10],
                                'class_weight': ['balanced'], # 
                            },
        "report": repD , 
    }
    prediction_list.append(predD)

#prediction_list = prediction_list[0:1]

repname = "1_from_all_23vars"
'''

# test writing of csv
try:
    fp = open(repname + ".csv", "w+")
except:
    sys.exit("Can't (over)write " + repname)
else:
    fp.close()


data_dict = make_data_dict(prediction_list)
print("read in", len(data_dict), "dataframes")

for p in prediction_list:

    logfname = p["pred_date"][0] + "_from_" +  p["model_name"] + ".txt"
    try:
        logfilehandle = open(logfname, 'a+')
    except Exception as e:
        sys.exit(e)
    printLog("\n", logfname)
    printLog(time.strftime('%H:%M:%S'))

    tune_model(p, data_dict)
    external_prediction(p, data_dict)
    graph_decision_trees(p, "pics")

    printLog(time.strftime('%H:%M:%S'))
    logfilehandle.close()

# define columns in report dict for output into csv file
cols = ["model_dates",
        "pred_date",
        "expl_vars",
        "resp_var",
        "n_estimators", 
        'max_features', 
        #"max_depth", 
        'min_samples_split',
        'min_samples_leaf', 
        "n_total",
        "n_training",
        "n_test",
        "acc_test",
        "acc_train",
        "acc_full",
        "acc_oob",
        "TP",
        "FP",
        "TN",
        "FN",
        "specificity",
        "sensitivity",
        "auc",
        "kappa",
        "var_importance", 
        "pred_date",
        "ext_acc",
        "ext_num_samples",
        "ext_healthy_prob",
        "ext_sds_prob",
        "ext_acc2",
        "ext_n_total",
        "ext_TP",
        "ext_TN",
        "ext_FN",
        "ext_FP",
        "ext_specificity",
        "ext_sensitivity",
        "ext_YourdansJ",
]

header = ", ".join([c for c in cols])
with open(repname+".csv", "w+") as f:
    print(header, file=f)
    for p in prediction_list:
        r = results_as_csv(p, cols)
        print(r, file=f)


sys.exit("Done")

#----------------------------------------------------------------------------------

