import numpy as np


class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero initialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1  # single set of weights needed
        self.d = 2  # input space is 2D. easier to visualize
        self.weights = np.zeros(self.d+1)

        self.change = np.zeros((self.d + 1))
    
    @staticmethod
    def preprocess(input_x):
        """
        Preprocess the input any way you seem fit.
        """
        # mean = np.mean(input_x, axis=0)
        # std = np.std(input_x, axis=0)
        #
        # input_x = (input_x - mean) / std

        return input_x

    @staticmethod
    def sigmoid(x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        return 1/(1+np.exp(-x))

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        b = np.ones((input_x.shape[0], 1))
        Input_X = np.hstack([input_x, b])

        z = Input_X.dot(self.weights)
        yp = self.sigmoid(z)

        loss = (-input_y * np.log(yp) - (1 - input_y) * np.log(1 - yp)).mean()
        return loss

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        b = np.ones((input_x.shape[0], 1))
        Input_X = np.hstack([input_x, b])

        gradient = (1 / input_x.shape[0]) * (Input_X.T.dot(Input_X.dot(self.weights) - input_y))

        return gradient

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        new_change = learning_rate * grad + momentum * self.change
        self.weights -= new_change
        self.change = new_change

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        def map_func(x):
            if x < 0.5:
                return 0
            else:
                return 1

        b = np.ones((input_x.shape[0], 1))
        Input_X = np.hstack([input_x, b])

        prediction = self.sigmoid(Input_X.dot(self.weights))

        prediction_mapping = np.array(list(map(map_func, prediction)))

        return prediction_mapping


class LinearClassifier:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        We have given the default zero initialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 3  # 3 classes
        self.d = 4  # 4 dimensional features
        self.weights = np.zeros(self.d+1)

        self.change = np.zeros((self.d+1))
    
    @staticmethod
    def preprocess(train_x):
        """
        Preprocess the input any way you seem fit.
        """
        # mean = np.mean(train_x, axis=0)
        # std = np.std(train_x, axis=0)
        #
        # train_x = (train_x - mean) / std

        return train_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        pass

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        b = np.ones((input_x.shape[0], 1))
        Input_X = np.hstack([input_x, b])

        loss = np.sum(0.5*(np.power(Input_X.dot(self.weights) - input_y, 2)))
        return loss

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        b = np.ones((input_x.shape[0], 1))
        Input_X = np.hstack([input_x, b])

        gradient = (1/input_x.shape[0])*(Input_X.T.dot(Input_X.dot(self.weights) - input_y))

        return gradient

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        new_change = learning_rate * grad + momentum*self.change
        self.weights -= new_change
        self.change = new_change

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        def map_func(x):
            if x < 0.5:
                return 0
            elif x < 1.5:
                return 1
            else:
                return 2

        b = np.ones((input_x.shape[0], 1))
        Input_X = np.hstack([input_x, b])

        prediction = Input_X.dot(self.weights)

        prediction_mapping = np.array(list(map(map_func, prediction)))

        return prediction_mapping
