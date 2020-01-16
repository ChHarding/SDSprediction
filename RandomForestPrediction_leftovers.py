# force prediction and model dates to be the same
model_data_ = model_data.copy()
prediction_data_ = prediction_data.copy()
for m in model_data_:
  model_name = m[0]
  model_data = [m]
  for p in prediction_data_:
    prediction_data = [p]
    
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
        #printLog("model": model_date, column_names)
        d = import_featureclass_to_dataframe(model_fc, workspace, column_names, add_NDVI)
        #printLog(d.describe())
        df_date_list.append([model_date, d]) # store the date for each df

    # create tuned model and run preditions
    for p in prediction_data: # 
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

            if model_date == pred_date and no_same_date == True:
                removed_flag = True 
                continue # skip this model date a we don't want pred date in model
            
            model_dates.append(model_date)
            df_list.append(df)

        data = pd.concat(df_list, sort=False)
        printLog("Model", model_name, "is made from", model_dates, end='')
        printLog(" - prediction date", pred_date, "was removed from model dates") if removed_flag else printLog()
        printLog(data.describe())

        # divide BGR NIR by their max for normalize?

        # redefine expl vars as they were renamed and NDVI may have been added
        expl_vars = ['Blue', 'Green', 'Red', 'NIR', 'Rotation']
        if add_NDVI: expl_vars += ["NDVI"]
        resp_var = "SDS"

        printLog("Predicting", pred_date, ": ", resp_var , "from", expl_vars)
        X = data[expl_vars] # Explanatory variables
        y = data[resp_var]  # Response variable

        # Split dataset into training set and test set
        SPLIT_RND_SEED = 12345
        test_size = 0.3
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=test_size, 
                                                            random_state=SPLIT_RND_SEED) 
        printLog("Using", len(X_train), "quadrants for training,", len(y_test), "quadrants for testing")


        # optimize model
        model = RandomForestClassifier(n_jobs=-1, random_state=12345, verbose=2)

        # Important parameters to tune
        # n_estimators (“ntree” in R)
        # max_features(“mtry” in R)
        # min_sample_leaf (“nodesize” in R)

        # single date model
        grid = {'n_estimators': [5, 11, 15, 21], 
                'max_features': [2, 3, 4, 5],
                'max_depth': [3, 7, 10, 15, 20],
                'min_samples_leaf': [1, 2, 3],
                'min_samples_split': [5, 7, 9, 11],
                }
        '''
        # multi date model
        grid = {'n_estimators': [ 45, 50, 55, 60], 
                'max_features': [2, 3, 4],
                'max_depth': [10, 15, 20, 25],
                'min_samples_leaf': [1, 2, 3],
                #'min_samples_split': [2, 3, 5, 7],
                }
        '''

        rf_gridsearch = GridSearchCV(estimator=model, 
                                    param_grid=grid, 
                                    scoring='roc_auc',
                                    n_jobs=-1,
                                    cv=5, 
                                    verbose=0, 
                                    return_train_score=True)
        rf_gridsearch.fit(X_train, y_train)
        #printLog("tuning done:", rf_gridsearch.best_params_)
        best_params = rf_gridsearch.best_params_

        # create optimized model
        rf = RandomForestClassifier(**best_params, 
                                oob_score=True, 
                                random_state=12345, 
                                verbose=False)

        # Train the tuned model using the training sets
        c = rf.fit(X_train, y_train)
        printLog("Tuned model:", c)


        # Names of explanatory variables in the file!
        expl_vars = ['MEAN_1', 'MEAN_2', 'MEAN_3', 'MEAN_4', 'Rotation'] 
        resp_vars = ['SDS'] # name of response Variable

        # list with all variables
        allVars = expl_vars + resp_vars
        column_names = ["Quadrat"] + allVars 


        # load verification (predition) data
        printLog("featureclass for prediction:", workspace, pred_fc,"\n", column_names)
        vdata = import_featureclass_to_dataframe(pred_fc, workspace, column_names, add_NDVI)



        # make a new report
        report_fn = "report_predict" + pred_date + "_from_" + model_name + ".txt"
        with open(report_fn, "w+") as rep:
            m_dates = [m[0] for m in model_data]
            printLog("Report for predicting", pred_date, "from", model_dates, file=rep)

            printLog("Model name:", model_name, file=rep)
            printLog("Note: prediction date", pred_date, "was removed from model dates", file=rep) if removed_flag else printLog()

            printLog("\nPredicting", resp_var , "from", expl_vars,  file=rep)

            '''
            printLog("\nCount of Rotation type for all quadrats:\n", 
                    data.groupby("Rotation")["Rotation"].count(), file=rep)

            printLog("\nCount by Rotation and SDS:\n", 
                    data.groupby(["Rotation", "SDS"])["Rotation"].count(), file=rep)

            printLog("\nCount by SDS and Rotation:\n", 
                    vdata.groupby(["SDS", "Rotation"])["SDS"].count(), file=rep)

            pd.options.display.float_format = '{:.2f}'.format 
            printLog("\nMeans by SDS and Rotation:\n", vdata.groupby(["SDS", "Rotation"])
                ["Blue","Red", "Green", "NIR"].mean().dropna(), file=rep)

            printLog("\nMeans by Rotation and SDS:\n", vdata.groupby(["Rotation", "SDS"])
                ["Blue","Red", "Green", "NIR"].mean().dropna(),  file=rep)
            pd.options.display.float_format = None
            '''



            printLog("\nUsing", len(X_train), "quadrants for training", 1-test_size, ",", len(y_test), "quadrants for testing", test_size, file=rep)
            printLog("grid search parameter list", grid,  file=rep)
            printLog("parameters for optimized model (scored by roc_auc)", best_params,  file=rep)
            printLog(c, file=rep)

            printLog('\nAccuracy on the training subset: {:.3f}'.format(rf.score(X_train, y_train)), file=rep)
            printLog('Accuracy on the test subset: {:.3f}'.format(rf.score(X_test, y_test)), file=rep)

            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            printLog(f'\nOut-of-bag score estimate: {rf.oob_score_:.3}', file=rep)
            
            printLog('\nConfusion (error) matrix of prediction', file=rep)
            cm = PD.DataFrame(confusion_matrix(y_test, y_pred))
            printLog(cm,  file=rep)

            stats = classification_report(y_test, y_pred,
                                labels=None,
                                target_names=["Healty", "SDS"],
                                sample_weight=None,
                                digits=2,
                                output_dict=False)
            printLog("\nClassification report:", file=rep)
            printLog(stats, file=rep)

            probs = rf.predict_proba(X_test)
            probs = probs[:, 1]
            auc = roc_auc_score(y_test, probs)
            printLog('\nReceiver Operator Characteristic (ROC) curve, AUC: %.3f' % auc, file=rep)

            cohen_score = cohen_kappa_score(y_test, y_pred)
            printLog("\nKappa score:", cohen_score, file=rep)

            printLog("\nVariable importance:", file=rep)
            fi = PD.DataFrame({'variable name': list(X_test.columns),
                    'importance': rf.feature_importances_})
            printLog(fi.sort_values('importance', ascending = False), file=rep)

        # Prediction
        pred_dict = {}# dict for (cumulative) predictions per quadrat id
        num_trees = len(rf.estimators_)

        # loop over all trees
        for ti,t in enumerate(rf.estimators_):
            #num_rows = len(vdata.index)
                        
            for i in vdata.index:
                q = vdata.loc[i] # get each quadrat as a Series
                #printLog(q)
                qid = q["Quadrat"] # get id for later
                
                # remove non exploratory variables
                del q["Quadrat"]
                del q["SDS"]
                #printLog(q)
                
                # predict SDS for this quadrat and store in dict
                p =  predict(t, q)
            
                # if we have no entry for this qid yet, init with 0
                if pred_dict.get(qid) == None:  
                    pred_dict[qid] = 0
                    
                # add 1 / num_trees to this qid, so we get a probability (0.0  - 1.0) in the end
                pred_dict[qid] += p / num_trees

            printLog('Predicting using tree', ti, "of", num_trees)

        # write the prediction for each quadrat into the pred column of vdata
        for i in vdata.index:
            q = vdata.loc[i]
            qid = q["Quadrat"]
            SDS = q["SDS"]
            #printLog(qid, pred_dict[qid])
            pred_prob  = pred_dict[qid]
            vdata.at[i, 'pred_prob'] = pred_prob
            pred = 0
            
            if pred_prob > 0.5: pred = 1
                
            # set prediction type (True positive,    
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
            printLog("\n====================================================\nEstimation Statistics for predicting", 
            pred_date, "from", model_name, file=rep)
            tc1 = PD.DataFrame(confusion_matrix(vdata["pred"], vdata["SDS"]))
            numTrue = tc1.loc[0,0] + tc1.loc[1,1]
            numFalse =  tc1.loc[1,0] + tc1.loc[0,1]
            numTotal = numTrue + numFalse
            printLog("Accuracy:", numTrue, "predicted as True of", numTotal, "total = ", numTrue / numTotal, file=rep)
            printLog("\nTable of confusion:\n", tc1, file=rep)

            '''
            tc2 = vdata.groupby("pred_type")["pred_type"].count()
            printLog("\nCount of Rotation type for all quadrats:\n", 
                    data.groupby("Rotation")["Rotation"].count(), file=rep)

            printLog("\nCount for Prediction Type (Table of confusion):\n", tc2, file=rep)

            printLog("\nCount by Rotation and Prediction type:\n", 
                    vdata.groupby(["Rotation", "pred_type"])["Rotation"].count(), file=rep)

            printLog("\nCount by Prediction type and Rotation:\n", 
                    vdata.groupby(["pred_type", "Rotation"])["pred_type"].count(), file=rep)

            pd.options.display.float_format = '{:.2f}'.format 
            printLog("\nMeans by Prediction type and Rotation:\n", vdata.groupby(["pred_type", "Rotation"])
                ["Blue","Red", "Green", "NIR", "pred_prob"].mean().dropna(), file=rep)

            printLog("\nMeans by Rotation and Prediction type:\n", vdata.groupby(["Rotation", "pred_type"])
                ["Blue","Red", "Green", "NIR", "pred_prob"].mean().dropna(), file=rep)
            pd.options.display.float_format = None
            '''
        
        '''
        # save quadrat id and prediction results in csv
        pred_res_table = pred_date + "_from_" + model_name + "_prediction_results_table.csv"
        cols = [ vdata[n] for n in ["Quadrat", "pred_prob", "pred", "pred_type" ] ]
        qpredres = PD.concat(cols, axis=1)
        #qpredres.head()
        qpredres.to_csv(pred_res_table)       

        # Copy feature class and join results table to it
        results_fc = "pred_" + pred_date + "_from_" + model_name
        printLog("Copying",  pred_fc, "to", results_fc)
        arcpy.env.overwriteOutput = True
        try:
            arcpy.Copy_management(pred_fc, results_fc)
        except Exception as e:
            printLog(e)
        else:
            printLog(arcpy.GetMessages())

        # also set the alias, otherwise it still looks like the pred layer name in the TOC
        try:
            arcpy.AlterAliasName(results_fc, results_fc)
        except Exception as e:
            printLog(e)
        else:
            printLog(arcpy.GetMessages())

        printLog("joining", pred_res_table, "to", results_fc)
        key = "Quadrat"
        try:
            arcpy.JoinField_management(results_fc, 
                                    key, 
                                    pred_res_table, 
                                    key)
        except Exception as e:
            printLog(e)
        else:
            # Show name and type of all fields
            printLog("Joined layer has these fields:")
            for field in arcpy.ListFields(results_fc):
                printLog("\t", field.name, " type:", field.type)

        '''

printLog("\nDone")
    