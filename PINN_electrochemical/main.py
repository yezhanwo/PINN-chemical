import numpy as np
import pybamm
from click.core import batch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from param.Param_02 import get_parameter_values


#电化学模型部分
# 参数设置
param=pybamm.ParameterValues(get_parameter_values())

# 实验设置
experiment=pybamm.Experiment([

                "Discharge at 0.1C until 2.7V",
    ])
# 求解和网格
solver=pybamm.CasadiSolver(extra_options_setup={"max_step_size":1})
var_pts={
    "x_n":25,
    "x_s":20,
    "x_p":25,
    "r_n":25,
    "r_p":25,
}
model=pybamm.lithium_ion.DFN()
model.variables["Negative average concentration [mol.m-3]"]=pybamm.x_average(model.variables["Negative particle concentration [mol.m-3]"])
sim=pybamm.Simulation(model,parameter_values=param,experiment=experiment,var_pts=var_pts,solver=solver)
sol=sim.solve([0,3600],initial_soc=1)
c_s_n=sol["Negative average concentration [mol.m-3]"].entries
c_s_n_avg=c_s_n.mean(axis=0)#求平均值，维度为0因为是对时间维度求平均，第二维度是空间维度就是一开始定义的x_n
# 变量导出，数据
time_step=sol["Time [s]"].entries#entries用来提取我要的具体数据，比如电压就能提取电压数字出来，否则还可能包括其他东西比如每个电压对应的特征点等
voltage=sol["Terminal voltage [V]"].entries
current=sol["Current [A]"].entries
# sim.plot()测试这部分没啥问题

#LSTM部分
import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# 数据处理和转换张量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def data_prepare(current,c_s_n_avg,voltage,len_sequence):
    features=np.column_stack((current,c_s_n_avg))#注意使用双括号，输入必须是元组，虽然意义就是两个一维被堆叠为一个而二维

    scaler_1=MinMaxScaler()
    features_scaler=scaler_1.fit_transform(features)

    scaler_2=MinMaxScaler()
    voltage_scaler=scaler_2.fit_transform(voltage.reshape(-1,1))#transform必须输入维度为二维（样本数，特征数），所以必须reshape，由（100，）变为（100，1）

    X,y=[],[]
    for i in range(len(features)-len_sequence):
        X.append(features_scaler[i:i+len_sequence])
        y.append(voltage_scaler[i+len_sequence])

    X=np.array(X)#本来矩阵每个元素是10个的2个特征（0.2，0.1），有100行，转变后变为（100，10，2）
    y=np.array(y)
    return  X,y,scaler_2#返回一个scaler方便后面反归一化

class Basedataset(Dataset): #class里为啥有东西，说明这个类是和torch里的Dataset类的功能差不多一致，可以直接继承过来使用
    def __init__(self,X,y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 模型搭建
class LstmModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):#这个顺序不可以乱排
        super().__init__()
        self.hiden_size=hidden_size
        self.num_layers=num_layers

        self.lstm=nn.LSTM(input_size,hidden_size, num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)#定义一个全连接层，处理lstm层输出的隐藏状态变为最后输出的目标值

    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hiden_size).to(x.device)#x size=(batch_size, sequence_length, hidden_size)。
        #这句话意思为所有堆叠层，所有批次，说有隐藏层都归0
        c0=torch.zeros(self.num_layers,x.size(0),self.hiden_size).to(x.device)

        out, _ =self.lstm(x, (h0,c0))
        #LSTM层的返回值是两个对象：所有时间步的输出和最终的隐藏状态与细胞状态。代码中的 _  是为了 忽略不需要的部分,即最后一个时间步的隐藏层和细胞状态
        out=self.fc(out[:,-1,:])
        return out

# 主函数定义和模型训练
def main():
    #使用上面定义的函数和类呗，分别先对数据进行处理分为X，y，
    X,y,scaler_target=data_prepare(current,c_s_n_avg,voltage,len_sequence=32)

    # 再进行训练集测试集拆分并转换为张量，
    X_train,X_test=train_test_split(X,test_size=0.2,shuffle=False)#LSTM可不能打乱哦
    y_train,y_test = train_test_split(y, test_size=0.2, shuffle=False)

    #用数据加载器把数据加载进去，这个有torch自己的定义
    train_data=Basedataset(X_train,y_train)
    test_data=Basedataset(X_test,y_test)

    batch_size=32
    test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False)
    train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=False)

    # 之后定义并训练模型
    input_size=X.shape[2]
    output_size=1
    num_layers=2
    hidden_size=64

    model=LstmModel(input_size,hidden_size,num_layers,output_size).to(device)#这一步只是搭建模型框架，所以还不需要输入X
    #损失函数和优化器
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    #开始训练！
    epochs=100
    for epoch in range(epochs):
        model.train()
        train_loss=0
        for batch_X,batch_y in train_loader:#这个应该也是固定搭配，Dataloader里有batch这个返回值
            batch_X,batch_y=batch_X.to(device),batch_y.to(device)
            out=model(batch_X)#这个时候就可以输入了真的开始训练了
            loss=criterion(out,batch_y)

            #固定搭配反向传播和优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()#item用来把张量化为标量

        # 模型预测
        model.eval()#切换到评估模式
        test_loss=0
        with torch.no_grad():#临时关闭梯度计算
            for batch_X,batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                out=model(batch_X)
                loss=criterion(batch_y,out)
                test_loss+=loss.item()
                #print中f允许在大括号中嵌入变量,.6f保留6位小数
        print(f'Epoch[{epoch}/{epochs}],train_loss[{train_loss/len(train_loader):.6f}],test_loss[{test_loss/len(test_loader):.6f}]')

    model.eval()
    with torch.no_grad():
        test_X=torch.tensor(X_test,dtype=torch.float32).to(device)#为啥要再张量变换一次？因为之前变化的张量可不是X，而是x，y一起打包变换的，不合用
        pre_y=model(test_X).cpu().numpy()
        y_pred=scaler_target.inverse_transform(pre_y)
        y_actual=scaler_target.inverse_transform(y_test)

        plt.figure()
        plt.plot(y_actual,label='actual')
        plt.plot(y_pred,label='predict')
        plt.xlabel('time')
        plt.ylabel('Volatge')
        plt.legend()
        plt.show()



if __name__=='__main__':
    main()



