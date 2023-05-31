import numpy as np
import pandas as pd

#建立支持向量机类，可传入的参数有数据集，学习率，正则化系数，最大迭代次数
#运行时在程序最后更改参数即可
#调用iter()方法进行子梯度下降的迭代，迭代次数为传入的数据
#调用show()方法返回迭代后的w，b以及各种评估数据
class Support_Vector_Machine:
    def __init__(self, dataset,learning_rate, C, max_iter):
        self.dataset = dataset
        self.row = self.dataset.shape[0]
        self.trainset = self.dataset.iloc[0:round(4 / 5 * self.row)]
        self.testset = self.dataset.iloc[round(4 / 5 * self.row):]
        self.column = self.dataset.shape[1]
        self.la = 1/C
        self.b = 0
        self.max_iter = max_iter
        self.w = np.zeros(self.column-1)+1
        self.lr = learning_rate

    def iter(self):
        iter = 1
        while iter<= self.max_iter:
            sum_w = np.zeros(self.column-1)
            sum_b = 0
            for i in range(round(4 / 5 * self.row)):
                y = self.trainset.iloc[i:i+1,self.column-1].values[0]
                if y == 0:
                    y = -1
                x = self.trainset.iloc[i:i+1,0:self.column-1].values[0]
                if y*np.dot(self.w,x)<1:
                    sum_w += y*x
                    sum_b += y
            sub_gradient_w = self.la*self.w-(sum_w/round(4 / 5 * self.row))
            sub_gradient_b = -(sum_b/round(4 / 5 * self.row))
            self.w =self.w - self.lr*sub_gradient_w
            self.b =self.b - self.lr*sub_gradient_b
            iter += 1

    def show(self):
        print(self.w,self.b)
        TP,FP,TN,FN = 0,0,0,0
        for i in range(self.row-round(4 / 5 * self.row)):
            y = self.testset.iloc[i:i + 1, self.column - 1].values[0]
            if y == 0:
                y = -1
            x = self.testset.iloc[i:i + 1, 0:self.column - 1].values[0]
            y_fitted = np.sign(np.dot(self.w,x)+self.b)
            if y == 1 and y_fitted == 1:
                TP += 1
            if y == 1 and y_fitted == -1:
                FN += 1
            if y == -1 and y_fitted == 1:
                FP += 1
            if y == -1 and y_fitted == -1:
                TN += 1
        print(TP,FP,TN,FN)
        print("accuraty",(TP+TN)/(self.row-round(4 / 5 * self.row)))
        print("precision",TP/(TP+FP))
        print("recall", TP / (TP + FN))




if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    model = Support_Vector_Machine(df, 0.03, 1, 70)
    model.iter()
    model.show()

