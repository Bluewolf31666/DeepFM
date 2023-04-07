from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

class Model():
    def Train_Model(df,sparse_features,dense_features,target):
        fixlen_feature_columns = [SparseFeat(feat, df[feat].max() + 1, embedding_dim=10) for feat in sparse_features]+[DenseFeat(feat, 1, )for feat in dense_features]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        ##
        train, test = train_test_split(df, test_size=0.2, random_state=2020)
        train_model_input = {name: train[name].values for name in feature_names}
        test_model_input = {name: test[name].values for name in feature_names}
        ##
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
        model.compile("adam", "mse", metrics=['mse'])
        history = model.fit(train_model_input, [train[target].values],batch_size=256, epochs=100, verbose=2, validation_split=0.2)
        pred_ans = model.predict(test_model_input)
        print("test mean_squared_error", round(mean_squared_error(test[target], pred_ans), 4))

        return(model)
    
    def Predict_Model(model,df2,sparse_features,dense_features,target):
        fixlen_feature_columns = [SparseFeat(feat, df2[feat].max() + 1, embedding_dim=10) for feat in sparse_features] +[DenseFeat(feat, 1, )
                                                                            for feat in dense_features]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        test_model_input2 = {name: df2[name].values for name in feature_names}
        result2=model.predict(test_model_input2)
        np_result=pd.Series(result2.round(2).flatten())
        return (np_result)
        

        