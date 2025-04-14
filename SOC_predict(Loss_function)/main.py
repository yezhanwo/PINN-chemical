import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import no_grad


df_fresh=pd.read_csv('Experimental_data_fresh_cell.csv')
df_aged=pd.read_csv('Experimental_data_aged_cell.csv')
df_OCV=pd.read_csv('OCV_vs_SOC_curve.csv')

#四分距法去除异常值，1.5为经验取值
def IQR_out(df,column):
    for col in column:
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3-Q1
        bound_low=Q1-1.5*IQR
        bound_high=Q3+1.5*IQR
        df[col]=np.clip(df[col],bound_low,bound_high)#clip函数的作用就是过了上下限就自动取边界值
    return df
df_fresh=IQR_out(df_fresh,['Current','Voltage','Temperature'])
df_aged=IQR_out(df_aged,['Current','Voltage','Temperature'])
df_OCV=IQR_out(df_OCV,['SOC','V0'])

#结合两个数据集，并在里面设置了索引来区分，同时把索引列调到第一列去了
df_combined=pd.concat([df_fresh,df_aged],keys=['fresh','aged'],names=['cell_type'])
df_combined = df_combined.reset_index(level=0)

df_train,df_test=train_test_split(df_combined,train_size=0.75,stratify=df_combined['cell_type'],random_state=42)

ocv_mapping = df_OCV.set_index('SOC')['V0']

def get_ocv(soc):
    return ocv_mapping.asof(soc)
inverse_OCV_SOC=df_OCV.set_index('V0')['SOC']
def get_SOC(voltage):
    return inverse_OCV_SOC.asof(voltage)#asof起返回小于目标索引最近的值，这里voltage就是个索引

#设置目标值
df_train['SOC']=df_train['Voltage'].apply(get_SOC)#apply就是一个能直接用函数的功能
df_test['SOC']=df_test['Voltage'].apply(get_SOC)

#归一化
features=['Voltage','Current','Temperature']
for feature in features:
    train_mean=df_train[feature].mean()
    test_mean=df_test[feature].mean()
    train_std=df_train[feature].std()
    test_std=df_test[feature].std()

    df_train[feature+'_scaled']=(df_train[feature]-train_mean)/train_std
    df_test[feature+'_scaled']=(df_test[feature]-test_mean)/test_std

import torch
import torch.nn as nn
#values用于将dataframe转换为numpy数组上
X_train=df_train[['Current_scaled','Voltage_scaled','Temperature_scaled']].values#单括号选中的是一列，双括号就相当于选中表格了
Y_train=df_train['SOC'].values.reshape(-1,1)
X_test=df_test[['Current_scaled','Voltage_scaled','Temperature_scaled']].values
Y_test=df_test['SOC'].values.reshape(-1,1)

X_train=torch.tensor(X_train,dtype=torch.float32)
Y_train=torch.tensor(Y_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
Y_test=torch.tensor(Y_test,dtype=torch.float32)

class PINN(nn.Module):
    def __init__(self):
        super(PINN,self).__init__()
        self.fc1=nn.Linear(3,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(32,1)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.relu(self.fc1(x))#定义全连接层必须过一遍非线性单元
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def physics_loss(self,input,output):
        current=input[:,1]
        capacity=1
        resistence=0.1

        soc_np=output.detach().numpy().flatten()
        for soc in soc_np:
            ocv=get_ocv(soc)
        ocv_tensor=torch.tensor(ocv,dtype=torch.float32).view(-1,1)

        dSOCdt=-current.view(-1,1)/capacity
        ocv_error=input[:,2]-(ocv_tensor+current.view(-1,1)*resistence)

        return torch.mean(torch.square(dSOCdt))+torch.mean(torch.square(ocv_error))

model=PINN()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epoches=20
batch_size=32
n=X_train.shape[0]
for epoch in range(epoches):
    rand=torch.randperm(n)
    for i in range(0,n,batch_size):
        indicies=rand[i:i+batch_size]
        X_batch=X_train[indicies]
        Y_batch=Y_train[indicies]

        optimizer.zero_grad()
        outputs=model(X_batch)
        mse_loss=criterion(outputs,Y_batch)
        physics_loss=model.physics_loss(X_batch,outputs)
        total_loss=mse_loss+physics_loss
        total_loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}/{epoches}   Loss {total_loss}')
with torch.no_grad():
    model.eval()
    Y_pred=model(X_test)
    pred_mse=criterion(Y_pred,Y_test)
    print(f'predict_loss={pred_mse}')
import matplotlib.pyplot as plt

# Use sample indices for the x-axis
sample_indices = range(len(Y_test))

plt.figure(figsize=(14, 6))
plt.plot(sample_indices, Y_test.numpy(), label='True SOC', color='blue', linewidth=1.5)
plt.plot(sample_indices, Y_pred.detach().numpy(), label='Predicted SOC', color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('SOC', fontsize=12)
plt.title('True vs Predicted SOC', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()




