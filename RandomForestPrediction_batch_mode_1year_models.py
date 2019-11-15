# Batch run making models and predicting them
import sys
import numpy as NUM
import numpy as np
import pandas as PD
import pandas as pd
import matplotlib.pyplot as PLOT
import matplotlib.pyplot as plt
import seaborn as SEA
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier # module to install is called scikit-learn
import eli5
import arcpy  

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from eli5.sklearn import PermutationImportance

from os.path import abspath

def import_featureclass_to_dataframe(feature_class, workspace, column_names, add_NDVI=True):
    arcpy.env.workspace = workspace
    fc_np = arcpy.da.FeatureClassToNumPyArray(feature_class, column_names)

    # Convert numpy array to Pandas dataframe
    data = PD.DataFrame(fc_np, columns=column_names)

    # use better names for the band numbers (NIR = Near Infrared)
    new_names = {'MEAN_1': 'Blue', 'MEAN_2': 'Green', 'MEAN_3': 'Red', 'MEAN_4': 'NIR'}
    data.rename(columns=new_names, inplace=True)

    if add_NDVI:
        # Calculate NDVI and put it in a new column
        ndvi = (data["NIR"] - data["Red"]) / (data["NIR"] + data["Red"])
        data.insert(5, 'NDVI', ndvi)


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


# MAIN
workspace = r"..\SDS_detection_ArcGISPro_project\SDS_detection_ArcGISPro_project.gdb" # must contain all featureclasses

# list of dates and feature class names for it
''' list of all dates, copy/paste the dates you want into the model and prediction lists below

    ["20160720", "Soybean_Quadrats_2016_zstats_cr_T20160720_161306_0e0e_3B_AnalyticMS_SR"], # OK
    ["20160821", "Soybean_Quadrats_2016_zstats_cr_T20160821_161512_0e26_3B_AnalyticMS_SR"], # OK
    ["20160831", "Soybean_Quadrats_2016_zstats_cr_T20160831_161520_0e20_3B_AnalyticMS_SR"], # OK

    ["20170720", "Soybean_Quadrats_2017_zstats_cr_T20170720_161916_101e_3B_AnalyticMS_SR"], # OK

    ["20180716", "Soybean_Quadrats_2018_zstats_cr_T20180716_163325_101b_3B_AnalyticMS_SR"], # OK
    ["20180822", "Soybean_Quadrats_2018_zstats_cr_T20180822_161812_0f46_3B_AnalyticMS_SR"], # OK
    ["20180724", "Soybean_Quadrats_2018_zstats_cr_T20180724_163356_1004_3B_AnalyticMS_SR"], # OK
    ["20180829", "Soybean_Quadrats_2018_zstats_cr_T20180829_163535_1021_3B_AnalyticMS_SR"], # OK
    ["20180831", "Soybean_Quadrats_2018_zstats_cr_T20180831_163515_1006_3B_AnalyticMS_SR"], # ok
'''

#
#  Configuration settings are next!
#
workspace = "SDS_detection_ArcGISPro_project.gdb" # folder of geoDB for shapefiles/feature classes

# model will be created and tuned from this/these date(s)
model_data = [

    ["20180716", "Soybean_Quadrats_2018_zstats_cr_T20180716_163325_101b_3B_AnalyticMS_SR"],  
    ["20180822", "Soybean_Quadrats_2018_zstats_cr_T20180822_161812_0f46_3B_AnalyticMS_SR"], 
    ["20180724", "Soybean_Quadrats_2018_zstats_cr_T20180724_163356_1004_3B_AnalyticMS_SR"], 
    ["20180829", "Soybean_Quadrats_2018_zstats_cr_T20180829_163535_1021_3B_AnalyticMS_SR"], 
    ["20180831", "Soybean_Quadrats_2018_zstats_cr_T20180831_163515_1006_3B_AnalyticMS_SR"],  
    ["20170720", "Soybean_Quadrats_2017_zstats_cr_T20170720_161916_101e_3B_AnalyticMS_SR"],

    ["20160720", "Soybean_Quadrats_2016_zstats_cr_T20160720_161306_0e0e_3B_AnalyticMS_SR"],
    ["20160821", "Soybean_Quadrats_2016_zstats_cr_T20160821_161512_0e26_3B_AnalyticMS_SR"],
    ["20160831", "Soybean_Quadrats_2016_zstats_cr_T20160831_161520_0e20_3B_AnalyticMS_SR"],
] 
model_name = "161718all_" # the name of fc and reports use this

# external predictions will be made from the model defined above for these dates
predition_data = [ 
    ["20160720", "Soybean_Quadrats_2016_zstats_cr_T20160720_161306_0e0e_3B_AnalyticMS_SR"], # OK
    ["20160821", "Soybean_Quadrats_2016_zstats_cr_T20160821_161512_0e26_3B_AnalyticMS_SR"], # OK
    ["20160831", "Soybean_Quadrats_2016_zstats_cr_T20160831_161520_0e20_3B_AnalyticMS_SR"], # OK

    ["20170720", "Soybean_Quadrats_2017_zstats_cr_T20170720_161916_101e_3B_AnalyticMS_SR"], # OK

    ["20180716", "Soybean_Quadrats_2018_zstats_cr_T20180716_163325_101b_3B_AnalyticMS_SR"], # OK
    ["20180822", "Soybean_Quadrats_2018_zstats_cr_T20180822_161812_0f46_3B_AnalyticMS_SR"], # OK
    ["20180724", "Soybean_Quadrats_2018_zstats_cr_T20180724_163356_1004_3B_AnalyticMS_SR"], # OK
    ["20180829", "Soybean_Quadrats_2018_zstats_cr_T20180829_163535_1021_3B_AnalyticMS_SR"], # OK
    ["20180831", "Soybean_Quadrats_2018_zstats_cr_T20180831_163515_1006_3B_AnalyticMS_SR"], # ok
]

# Names of explanatory variables as shown in the fc attribute table
expl_vars = ['MEAN_1', 'MEAN_2', 'MEAN_3', 'MEAN_4', 'Rotation'] 
resp_vars = ['SDS'] # name of response Variable

join_field = "Quadrat" # key for joining the prediction results back to the prediction fc
add_NDVI = False  # calculate NDVI for each date?


#
#  Done with configuration settings
#

info = {}  # dict for dicts with info about each prediction


# list with all variables
allVars = expl_vars + resp_vars
column_names = [join_field] + allVars 

# make dataframe from all feature classes
df_date_list = []
for m in model_data:  # make model from this
    model_fc = m[1]
    model_date = m[0]
    
    # load model data
    print(model_date, column_names)
    d = import_featureclass_to_dataframe(model_fc, workspace, column_names, add_NDVI)
    #print(d.describe())
    df_date_list.append([model_date, d]) # store the date for each df

# create tuned model and run preditions
for p in predition_data: # 
    pred_fc = p[1]
    pred_date = p[0]
    
    pi = {} # dict for prediction info
    pi["NDVI"] = "Y" if add_NDVI == True else "N"

    df_list = []
    model_dates = []
    removed_flag = False
    for m in df_date_list: 
        model_date = m[0]
        df = m[1]  # dataframe we already created for this date

        if model_date == pred_date:
            removed_flag = True 
            continue # skip this model date a we don't want pred date in model
        
        model_dates.append(model_date)
        df_list.append(df)

    data = pd.concat(df_list, sort=False)
    print("Model", model_name, "is made from", model_dates, end='')
    print(" - prediction date", pred_date, "was removed from model dates") if removed_flag else print()
    print(data.describe())

    # divide BGR NIR by there max for normalize?

    # redefine expl vars as they were renamed and NDVI may have been added
    expl_vars = ['Blue', 'Green', 'Red', 'NIR', 'Rotation']
    if add_NDVI: expl_vars += ["NDVI"]
    resp_var = "SDS"

    print("Predicting", resp_var , "from", expl_vars)
    X = data[expl_vars] # Explanatory variables
    y = data[resp_var]  # Response variable

    # Split dataset into training set and test set
    SPLIT_RND_SEED = 12345
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=SPLIT_RND_SEED) 
    print("Using", len(X_train), "quadrants for training,", len(y_test), "quadrants for testing")


    # optimize model
    model = RandomForestClassifier(n_jobs=-1, random_state=12345, verbose=2)

    # Important parameters to tune
    # n_estimators (“ntree” in R)
    # max_features(“mtry” in R)
    # min_sample_leaf (“nodesize” in R)

    grid = {'n_estimators': [ 45, 50, 55, 60],
            'max_features': [2, 3, 4],
            'max_depth': [10, 15, 20, 25],
            'min_samples_leaf': [1, 2, 3],
            #'min_samples_split': [2, 3, 5, 7],
            }
    
    # shut up GridSearchCV spam
    f = open('nul', 'w')
    old_stdout = sys.stdout
    print(old_stdout)
    sys.stderr = sys.stdout = f
    
    rf_gridsearch = GridSearchCV(estimator=model, 
                                param_grid=grid, 
                                scoring='roc_auc',
                                n_jobs=-1,
                                cv=5, 
                                verbose=0, 
                                return_train_score=True)
    rf_gridsearch.fit(X_train, y_train)

    best_params = rf_gridsearch.best_params_
    sys.stdout = old_stdout
    print("tuning done")



    # create optimized model
    rf = RandomForestClassifier(**best_params, 
                            oob_score=True, 
                            random_state=12345, 
                            verbose=False)

    # Train the tuned model using the training sets
    c = rf.fit(X_train, y_train)
    print(c)





    # Names of explanatory variables in the file!
    expl_vars = ['MEAN_1', 'MEAN_2', 'MEAN_3', 'MEAN_4', 'Rotation'] 
    resp_vars = ['SDS'] # name of response Variable

    # list with all variables
    allVars = expl_vars + resp_vars
    column_names = ["Quadrat"] + allVars 

    workspace = "SDS_detection_ArcGISPro_project.gdb"

    # load verification (predition) data
    print("featureclass for prediction:", workspace, pred_fc,"\n", column_names)
    vdata = import_featureclass_to_dataframe(pred_fc, workspace, column_names, add_NDVI)



    # make a new report
    report_fn = "report_predict" + pred_date + "_from_" + model_name + ".txt"
    with open(report_fn, "w+") as rep:
        m_dates = [m[0] for m in model_data]
        print("Report for predicting", pred_date, "from", model_dates, file=rep)

        print("Model name:", model_name, file=rep)
        print("Note: prediction date", pred_date, "was removed from model dates", file=rep) if removed_flag else print()

        print("\nPredicting", resp_var , "from", expl_vars,  file=rep)

        print("\nCount of Rotation type for all quadrats:\n", 
                data.groupby("Rotation")["Rotation"].count(), file=rep)

        print("\nCount by Rotation and SDS:\n", 
                data.groupby(["Rotation", "SDS"])["Rotation"].count(), file=rep)

        print("\nCount by SDS and Rotation:\n", 
                vdata.groupby(["SDS", "Rotation"])["SDS"].count(), file=rep)

        pd.options.display.float_format = '{:.2f}'.format 
        print("\nMeans by SDS and Rotation:\n", vdata.groupby(["SDS", "Rotation"])
            ["Blue","Red", "Green", "NIR"].mean().dropna(), file=rep)

        print("\nMeans by Rotation and SDS:\n", vdata.groupby(["Rotation", "SDS"])
            ["Blue","Red", "Green", "NIR"].mean().dropna(),  file=rep)
        pd.options.display.float_format = None




        print("\nUsing", len(X_train), "quadrants for training", 1-test_size, ",", len(y_test), "quadrants for testing", test_size, file=rep)
        print("parameters for optimized model (scored by roc_auc)", best_params,  file=rep)
        print(c, file=rep)

        print('\nAccuracy on the training subset: {:.3f}'.format(rf.score(X_train, y_train)), file=rep)
        print('Accuracy on the test subset: {:.3f}'.format(rf.score(X_test, y_test)), file=rep)

        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'\nOut-of-bag score estimate: {rf.oob_score_:.3}', file=rep)
        
        print('\nConfusion (error) matrix of prediction', file=rep)
        cm = PD.DataFrame(confusion_matrix(y_test, y_pred))
        print(cm,  file=rep)

        stats = classification_report(y_test, y_pred,
                              labels=None,
                              target_names=["Healty", "SDS"],
                              sample_weight=None,
                              digits=2,
                              output_dict=False)
        print("\nClassification report:", file=rep)
        print(stats, file=rep)

        probs = rf.predict_proba(X_test)
        probs = probs[:, 1]
        auc = roc_auc_score(y_test, probs)
        print('\nReceiver Operator Characteristic (ROC) curve, AUC: %.3f' % auc, file=rep)

        cohen_score = cohen_kappa_score(y_test, y_pred)
        print("\nKappa score:", cohen_score, file=rep)

        print("\nVariable importance:", file=rep)
        fi = PD.DataFrame({'variable name': list(X_test.columns),
                   'importance': rf.feature_importances_})
        print(fi.sort_values('importance', ascending = False), file=rep)

    # Prediction
    pred_dict = {}# dict for (cumulative) predictions per quadrat id
    num_trees = len(rf.estimators_)

    # loop over all trees
    for ti,t in enumerate(rf.estimators_):
        #num_rows = len(vdata.index)
                    
        for i in vdata.index:
            q = vdata.loc[i] # get each quadrat as a Series
            #print(q)
            qid = q["Quadrat"] # get id for later
            
            # remove non exploratory variables
            del q["Quadrat"]
            del q["SDS"]
            #print(q)
            
            # predict SDS for this quadrat and store in dict
            p =  predict(t, q)
        
            # if we have no entry for this qid yet, init with 0
            if pred_dict.get(qid) == None:  
                pred_dict[qid] = 0
                
            # add 1 / num_trees to this qid, so we get a probability (0.0  - 1.0) in the end
            pred_dict[qid] += p / num_trees

        print('Predicted using tree', ti, "of", num_trees)

    # write the prediction for each quadrat into the pred column of vdata
    for i in vdata.index:
        q = vdata.loc[i]
        qid = q["Quadrat"]
        SDS = q["SDS"]
        #print(qid, pred_dict[qid])
        pred_prob  = pred_dict[qid]
        vdata.at[i, 'pred_prob'] = pred_prob
        pred = 0
        
        # Only accept prediction if prob is 0.0 - 0.33 (0) or 0.66 - 1.0 (1)
        if pred_prob > 0.33 and pred_prob < 0.66:
            pred_type = "NA"
            pred = -1
        else:
            if pred_prob > 0.5: pred = 1
            
                
            if SDS == pred:
                if SDS == 1:
                    pred_type = "TP" # true positive
                else:
                    pred_type = "TN" # true negative
            else:
                if SDS == 1:
                    pred_type = "FN" # true positive
                else:
                    pred_type = "FP" # true negative
            
        vdata.at[i, 'pred'] = pred
        vdata.at[i, 'pred_type'] = pred_type
        
    vdata['pred'] = vdata['pred'].astype('int')


    with open(report_fn, "a+") as rep:
        print("\n====================================================\nEstimation Statistics for predicting", 
        pred_date, "from", model_name, file=rep)
        tc1 = PD.DataFrame(confusion_matrix(vdata["pred"], vdata["SDS"]))
        numTrue = tc1.loc[1,1] + tc1.loc[2,2]
        numFalse =  tc1.loc[2,1] + tc1.loc[1,2]
        numNotNA = numTrue + numFalse
        numNA = len(vdata.index) - numNotNA
        print("Accuracy:", numTrue, "correct of", numNotNA, " notNA predictions:", numTrue / numNotNA, file=rep)
        print(numNA, "predictions are between 0.33 and 0.66 probability (=> NA)", file=rep)
        tc2 = vdata.groupby("pred_type")["pred_type"].count()
        #print("\nTable of confusion:\n", tc1, file=rep)

        print("\nCount of Rotation type for all quadrats:\n", 
                data.groupby("Rotation")["Rotation"].count(), file=rep)

        print("\nCount for Prediction Type (Table of confusion):\n", tc2, file=rep)

        print("\nCount by Rotation and Prediction type:\n", 
                vdata.groupby(["Rotation", "pred_type"])["Rotation"].count(), file=rep)

        print("\nCount by Prediction type and Rotation:\n", 
                vdata.groupby(["pred_type", "Rotation"])["pred_type"].count(), file=rep)

        pd.options.display.float_format = '{:.2f}'.format 
        print("\nMeans by Prediction type and Rotation:\n", vdata.groupby(["pred_type", "Rotation"])
            ["Blue","Red", "Green", "NIR", "pred_prob"].mean().dropna(), file=rep)

        print("\nMeans by Rotation and Prediction type:\n", vdata.groupby(["Rotation", "pred_type"])
            ["Blue","Red", "Green", "NIR", "pred_prob"].mean().dropna(), file=rep)
        pd.options.display.float_format = None

    
    # save quadrat id and prediction results in csv
    pred_res_table = pred_date + "_from_" + model_name + "_prediction_results_table.csv"
    cols = [ vdata[n] for n in ["Quadrat", "pred_prob", "pred", "pred_type" ] ]
    qpredres = PD.concat(cols, axis=1)
    #qpredres.head()
    qpredres.to_csv(pred_res_table)       

    # Copy feature class and join results table to it
    results_fc = "pred_" + pred_date + "_from_" + model_name
    print("Copying",  pred_fc, "to", results_fc)
    arcpy.env.overwriteOutput = True
    try:
        arcpy.Copy_management(pred_fc, results_fc)
    except Exception as e:
        print(e)
    else:
        print(arcpy.GetMessages())

    # also set the alias, otherwise it still looks like the pred layer name in the TOC
    try:
        arcpy.AlterAliasName(results_fc, results_fc)
    except Exception as e:
        print(e)
    else:
        print(arcpy.GetMessages())

    print("joining", pred_res_table, "to", results_fc)
    key = "Quadrat"
    try:
        arcpy.JoinField_management(results_fc, 
                                key, 
                                pred_res_table, 
                                key)
    except Exception as e:
        print(e)
    else:
        # Show name and type of all fields
        print("Joined layer has these fields:")
        for field in arcpy.ListFields(results_fc):
            print("\t", field.name, " type:", field.type)

    
    print("\nDone")