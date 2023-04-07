from preprocessing import dataset
from model.DeepFM import Model
df,df2,sparse_features,dense_features,target,prediction= dataset.computer()
learning_model=Model.Train_Model(df,sparse_features,dense_features,target)
np_result=Model.Predict_Model(learning_model,df2,sparse_features,dense_features,target)
prediction['rating']=np_result
print(prediction)
