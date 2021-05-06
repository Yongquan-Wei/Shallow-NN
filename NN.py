import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
"""
#读取数据并显示出散点图
X, Y = load_planar_dataset()#X.shaoe=(2,400),Y.shape=(1,400)
plt.scatter(X[0], X[1], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
plt.show()
#试试逻辑回归,显然它只能拟合一条线性的决策边界
Y_=Y.reshape(-1)
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y_)
plot_decision_boundary(lambda x: clf.predict(x), X, Y) #绘制决策边界
plt.title("Logistic Regression") #图标题
plt.show()
LR_predictions  = clf.predict(X.T) #预测结果
print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) + 
		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       "% " + "(正确标记的数据点所占的百分比)")

"""
#使用单隐层的神经网络
#超参数
LR=1
iterations=10000
class NN():
    def __init__(self,n1,n2,n3,g):
        """
        参数：
            x=输入节点数
            y=输出节点数
            n=隐藏层节点数
            g=激活函数
        """
        self.n1=n1
        self.n2=n2
        self.n3=n3
        #参数矩阵
        self.W1=np.random.randn(self.n2,self.n1)*0.001
        self.b1=np.zeros((self.n2,1))
        self.W2=np.random.randn(self.n3,self.n2)*0.001
        self.b2=np.zeros((self.n3,1))
        self.g=g
        #激活函数
        if g=='tanh':
            self.func=lambda z:(np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif g=='Relu':
            self.func=lambda z:np.where(z<0,0,z)
            self.d=lambda x:np.where(x<0,0,1)
        elif g=='leaky Relu':
            self.func=lambda z:np.where(z<0,0.01*z,z)
            self.d=lambda x:np.where(x<0,0.01,1)
        elif g=='sigmoid':
            self.func=lambda z:.5 * (1 + np.tanh(.5 * z))

        
    def forward(self,X):
        Z1=np.dot(self.W1,X)+self.b1
        A1=self.func(Z1)
        Z2=np.dot(self.W2,A1)+self.b2
        A2=.5 * (1 + np.tanh(.5 * Z2))#输出层使用sigmoid输出二元分类概率
        cache={
            "Z1":Z1,
            "A1":A1,
            "Z2":Z2,
            "A2":A2
        }
        return A2,cache
    
    def loss(self,A2,Y):
        logprobs=np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        assert(logprobs.shape[0]==1)
        cost = - np.sum(logprobs) / logprobs.shape[1]
        cost = float(np.squeeze(cost))
        return cost

    def backward(self,cache,X,Y):
        m = X.shape[1]
        A1=cache["A1"]
        A2=cache["A2"]
        Z1=cache["Z1"]
        #计算梯度
        dZ2=A2-Y#这是交叉熵的第一次求导，sigmoid的导数
        dW2=(1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)#保证还有两个维度
        #各激活函数的dZ1不同
        if self.g=='tanh':
            dZ1 = np.multiply(np.dot(self.W2.T, dZ2), 1 - np.power(A1, 2))
        elif self.g=='Relu':
            dZ1 = np.multiply(np.dot(self.W2.T, dZ2), self.d(Z1))
        elif self.g=='leaky Relu':
            dZ1 = np.multiply(np.dot(self.W2.T, dZ2), self.d(Z1))
        elif self.g=='sigmoid':
            dZ1 = np.multiply(np.dot(self.W2.T, dZ2), np.multiply(A1,1-A1))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2 }
        return grads

    def optimize(self,grads):
        dW1,dW2 = grads["dW1"],grads["dW2"]
        db1,db2 = grads["db1"],grads["db2"]
        self.W1 = self.W1 - LR * dW1
        self.b1 = self.b1 - LR * db1
        self.W2 = self.W2 - LR * dW2
        self.b2 = self.b2 - LR * db2
    
    def predict(self,X):
        Z1=np.dot(self.W1,X)+self.b1
        A1=self.func(Z1)
        Z2=np.dot(self.W2,A1)+self.b2
        A2=.5 * (1 + np.tanh(.5 * Z2))#输出层使用sigmoid输出二元分类概率
        predictions=np.round(A2)
        return predictions

def main():
    #default Dataset
    #X, Y = load_planar_dataset()
    #extra Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}
    dataset = "blobs"
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    # make blobs binary
    if dataset == "blobs":
        Y = Y % 2
    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    #确定神经网络结构
    n1=X.shape[0]
    n2=5
    n3=Y.shape[0]
    assert(n3==1 and n1==2)
    net=NN(n1,n2,n3,'leaky Relu')
    costs=[]
    for i in range(iterations):
        A2,cache=net.forward(X)
        cost=net.loss(A2,Y)
        costs.append(cost)
        grads=net.backward(cache,X,Y)
        net.optimize(grads)
    plt.plot(costs)
    plt.show()
    plot_decision_boundary(lambda x: net.predict(x.T), X, Y)
    plt.show()
    predictions = net.predict(X)
    print ('神经网络的准确性: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')


if __name__ == "__main__":
    main()













    
    


