import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from deepctr.feature_column import SparseFeat, DenseFeat
from sklearn.preprocessing import MinMaxScaler
import Setting

class dataset():
    def computer():
        df=pd.read_csv(Setting.data_address)

        ram = []
        for i in df['ram']:
            ram.append(int(i.split(' ')[0]))
            
        df['ram_volume'] = ram

        ram_type = []
        for i in df['ram']:
            ram_type.append(int(i.split(' ')[0]))    
        df['ram_type'] = ram_type

        df=df.drop(['ram','img_link'],axis=1)

        cpu = []
        for i in df['processor']:
            cpu.append(str(i.split(' ')[0]))
            
        df['cpu_maker'] = cpu

        cpu_gen = []
        for i in df['processor']:
            if '(' in i:
                cpu_gen.append(int((str(i.split(' ')[-2]).replace("(","")).replace("th","").replace("rd","")))
            else:
                cpu_gen.append(0)
        #print(cpu_gen)
        df['cpu_generation'] = cpu_gen

        df=df.drop(['processor','name'],axis=1)

        df2=df[df['storage']!='PCI-e SSD (NVMe) ready,Silver-Lining Print Keyboard,Matrix Display (Extend),Cooler Boost 5,Hi-Res Audio,Nahimic 3,144Hz Panel,Thin Bezel,RGB Gaming Keyboard,Speaker Tuning Engine,MSI Center'].copy()
        df3=df2[df2['storage']!='PCI-e Gen4 SSD?SHIFT?Matrix Display (Extend)?Cooler Boost 3?Thunderbolt 4?Finger Print Security?True Color 2.0?Hi-Res Audio?Nahimic 3? 4-Sided Thin bezel?MSI Center?Silky Smooth Touchpad?Military-Grade Durability'].copy()
        df4=df3.drop(['no_of_ratings'],axis=1)
        df4=df4.drop('Unnamed: 0',axis=1)
        df4=df4.drop('no_of_reviews',axis=1)

        df4.rename(columns={'price(in Rs.)':'price','display(in inch)':'display'},inplace=True)

        prediction=df4[df4["rating"].isna()]
        data=df4[df4["rating"].notnull()]
        
        
        for i in [12.0, 13.5, 14.5, 15.3, 17.0]:
            prediction=prediction[prediction['display']!=i]
        for i in [3, 13, 6]:
            prediction=prediction[prediction['cpu_generation']!=i]
        prediction=prediction[prediction['storage']!='4 TB SSD']
        prediction=prediction.reset_index(drop=True)
        
        df = data.copy()
        sparse_features = ['os', 'storage', 'display', 'ram_volume', 'ram_type','cpu_maker', 'cpu_generation']
        dense_features =['price']
        target = ['rating']
        
        df2 = prediction.copy()
        sparse_features = ['os', 'storage', 'display', 'ram_volume', 'ram_type','cpu_maker', 'cpu_generation']
        dense_features =['price']
        target = 'rating'
        
        for feat in sparse_features:
            lbe = LabelEncoder()
            lbe.fit(df[feat])
            df[feat] = lbe.transform(df[feat])
            bin = pickle.dumps(lbe)

        mms = MinMaxScaler(feature_range=(0, 1))
        df[dense_features] = mms.fit_transform(df[dense_features])
        df2[dense_features] = mms.fit_transform(df2[dense_features])
        
        
        for feat in sparse_features:
            lbe = pickle.loads(bin)
            lbe.fit(df2[feat])
            df2[feat] = lbe.transform(df2[feat])
        
        return (df,df2,sparse_features,dense_features,target,prediction)




