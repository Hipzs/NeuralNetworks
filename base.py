import numpy as np


# I aim to create my own NN with basic library


# Step 1 :

# Create a basic dense NN

# => forward pass



class Dense():

    def softmax(self,x):
        exp = np.exp(x)
        return exp/exp.sum()

    def nada(self,x):
        return x

    def relu(self,x):
        return np.max(0,x)


    def __init__(self,input,output,activation='nada'):

        self.input = input
        self.output = output
        self.W = np.random.rand(self.input,self.output)

        act_list=['softmax','nada']
        if activation in act_list:
            self.activation = getattr(self,activation)
        else :
            self.activation = getattr(self,"nada")


    def forward(self,x):

        return self.activation(np.dot(x,self.W))



class Model():

    def MSE(self,x,y):
        return (1/y.shape[0])*np.sum(np.square(x-y))

    def max_log_likelihood(self,x,y):
        return -np.log(x)


    def __init__(self,loss = 'MSE'):
        self.Layers = []

        loss_list=['MSE','max_log_likelihood']
        if loss in loss_list:
            self.loss = getattr(self,loss)
        else :
            self.loss = getattr(self,"MSE")


    def add(self,layer):
        self.Layers.append(layer)


    def forward(self,x):

        for layer in self.Layers:
            x = layer.forward(x)

        return x


    def back(self,x,y):

        y_pred = self.forward(x)
        Loss = self.loss(y_pred ,y)








if __name__=="__main__":

    M = Model()
    a = Dense(3,2)
    b = Dense(2,1,activation='softmax')

    M.add(a)
    M.add(b)

    x=np.array([[4,0,4],[4,4,4]])
    y=np.array([1,0,0,0,0])
    y = np.array([[1],[0]])
    w=np.array([[1,1,1],[1,1,1]])

    #y =M.forward(x)
    M.back(x,y)
    #M.back(x,y)
