import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
    """
        Parameters:
        -----------
        n_iterations: int
            numbers of iteration for gradient descent
        learning_rate: float
            Learning rate of gradient descent
    """
    def __init__(self, learning_rate=.1, n_iterations=4000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def initialize_weights(self, n_features):
        # initialize parameter
        # range of parameter[-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        #insert value of b
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):

        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        # insert a row of feature x1 to X, x1=0, insert to 0th row
        #print('type x:',type(X))
        # matrix to nparray
        #X = np.array(X)
        X_cc = np.insert(X, 0, 1, axis=1)
        print(X_cc.shape)
        print(y.shape)
        y = np.reshape(y, (m_samples, 1))
        print(y.shape)

        #self.w = np.matrix(self.w)
        # training for gradient descent 
        for i in range(self.n_iterations):
            h_x = np.array(X_cc.dot(self.w),dtype=np.float32)
            #print('h_x',h_x.shape)
            y_pred = sigmoid(h_x)
            #print(y_pred.shape)
            #print(y.shape)
            #calculate gradient, use all data
            w_grad = X_cc.T.dot(y_pred - y)
            #print(‘gradient:’,w_grad)
            #update parameters
            self.w = self.w - self.learning_rate * w_grad

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        out = sigmoid(h_x)
        print('out:',out)
        y_pred = np.round(out)
        return y_pred.astype(int)
    
if __name__ == '__main__':
    x = np.array([[3,4,5],
         [4,6,7],
         [6,7,8]])
    y = np.array([2,3,4])
    
    print('vvv',x.shape)
    
    m = LogisticRegression()
    m.initialize_weights(x.shape[1])
    m.fit(x,y)
    predit = np.array([[3,5,6]])
    res = m.predict(predit)
    
    print(res)