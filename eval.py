def evaluate_predictions(solution_url, test_data_url, **kwargs):
    # if kwargs["competition_id"] == 1
    #   return competition_1_evaluation(**)

    # import the library.
    import pandas as pd
    import math

    # kwargs["test_data_url"]



    # Read data from url
    actual_data = pd.read_csv(kwargs["solution_url"]);
    test_data = pd.read_csv(kwargs["submission_file_url"]);

    # features
    no_of_features = kwargs["no_of_variables"]

    # cleaning old_data
    actual_data.columns = ['oid', 'actual'];  # set the new column names
    actual_data.sort_values('oid', inplace=True);
    actual_data = actual_data.set_index('oid');
    actual_data.index.name = None

    # cleaning new_data
    test_data.columns = ['pid', 'predicted'];  # set the new column names
    test_data.sort_values('pid', inplace=True);
    test_data = test_data.set_index('pid');
    test_data.index.name = None

    # extracting predictions
    actual_predictions = actual_data.ix[:, 0];
    new_predictions = test_data.ix[:, 0];
	
    #unique_labels
    labels = actual_predictions.unique()
    global TP,TN,FP,FN;
    #for binary classification
    if len(labels)==2:
        for i in range(len(actual_predictions)):
            if labels[1]==actual_predictions[actual_predictions.index[i]]:
                actual_predictions[actual_predictions.index[i]]=0;
            else:
                actual_predictions[actual_predictions.index[i]]=1;
                
        for i in range(len(new_predictions)):
            if labels[1]==new_predictions[new_predictions.index[i]]:
                new_predictions[new_predictions.index[i]]=0;
            else:
                new_predictions[new_predictions.index[i]]=1; 
                        
                
                
        TN = 0.0;
        TP = 0.0;
        FP = 0.0;
        FN = 0.0;
        
        for i in range(len(actual_predictions)):
            if (actual_predictions[actual_predictions.index[i]] -new_predictions[new_predictions.index[i]])==0:
                if actual_predictions[actual_predictions.index[i]]==0:
                    TN=TN+1;
                else:
                    TP=TP+1;
            else:
                if actual_predictions[actual_predictions.index[i]]==0:
                    FP=FP+1;
                else:
                    FN=FN+1;

    # return the Residualstandarderror for regression
    if kwargs["competition_id"] == '1000':
        return round(math.sqrt(sum((actual_predictions - new_predictions) ** 2) / (len(actual_predictions) - 2)), 3);

    #Multiple R-squared for regression
    elif kwargs["competition_id"] == '1001':
        return round (sum( (new_predictions - actual_predictions.mean() )**2 ) / sum( (actual_predictions -actual_predictions.mean() )**2),4);

    #Adjusted R-squared for regression
    elif kwargs["competition_id"] == '1002':
        r2=round (sum( (new_predictions - actual_predictions.mean() )**2 ) / sum( (actual_predictions -actual_predictions.mean() )**2),4);
        return round(1 - ((1-r2)*( len(actual_predictions)- 1))/(len(actual_predictions) -no_of_features-3) , 4 );

    #F-Statistic for regression
    elif kwargs["competition_id"] == '1003':
        msm = sum((new_predictions - (actual_predictions).mean())**2)/(features-1);
        mse = sum((actual_predictions - new_predictions)**2)/(len(actual_predictions) - (no_of_features+1));
        return msm/mse;

    #Accuaracy for binary classification
    elif kwargs["competition_id"] == '1004':
        return (TP+TN)/(TP+TN+FP+FN);
            
    #Recall for binary classification 
    elif kwargs["competition_id"] == '1005':
        return (TP)/(TP+FN);
        
    #F1-score for binary classification
    elif kwargs["competition_id"] == '1006':
        return 2/(((TP)/(TP+FN))+((TP)/(TP+FP)));
        
    #Precision for binary classification
    elif kwargs["competition_id"] == '1007':
        return (TP)/(TP+FP);
        
    #Specificity for binary classification
    elif kwargs["competition_id"] == '1008':
        return (TN)/(TN+FP);
        
    #Kappa for binary classification
    elif kwargs["competition_id"] == '1009':
        total=(TP+TN)/(TP+TN+FP+FN);
        Random=(((TP+FN)/(TP+TN+FP+FN))*((TP+FP)/(TP+TN+FP+FN)))+(((FP+TN)/(TP+TN+FP+FN))*((FN+TN)/(TP+TN+FP+FN)));
        return (total-Random)/(1-Random);

    #mean misclassification error
    elif kwargs["competition_id"] == '1010':
        count=0.0;
        i=0;
        for i in range(len(actual_predictions)):
            if (new_predictions[new_predictions.index[i]])!=actual_predictions[actual_predictions.index[i]]:
                count=count+1;

        return round(count/len(actual_predictions),4);

    #log - loss for multiclass classification
    #elif kwargs["competition_id"] == 1011:
    #    sum = 0.0;
    #    for i in range(len(actual_predictions)):
    #        sum=sum + math.log(test_data.ix[i+1,:][actual_predictions[i+1]]);
        
    #    return (-sum)/len(actual_predictions);
    

