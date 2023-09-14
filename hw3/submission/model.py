import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.gaussian = {}
        self.bernoulli = {}
        self.laplace = {}
        self.exponential = {}
        self.multinomial = {}

    def fit(self, x, y):

        """Start of your code."""
        """
        x : np.array of shape (n,10)
        y : np.array of shape (n,)
        Create a variable to store number of unique classes in the dataset.
        Assume Prior for each class to be ratio of number of data points in that class to total number of data points.
        Fit a distribution for each feature for each class.
        Store the parameters of the distribution in suitable data structure, for example you could create a class for each distribution and store the parameters in the class object.
        You can create a separate function for fitting each distribution in its and call it here.
        """

        """Start your code"""
        for label in np.unique(y).astype(int):
            self.priors[label] = np.bincount(y.astype(int))[label] / len(x)

            x_label = x[y == label]

            mean_x1 = np.mean(x_label.T[0])
            mean_x2 = np.mean(x_label.T[1])
            var_x1 = np.var(x_label.T[0])
            var_x2 = np.var(x_label.T[1])

            p_x3 = np.mean(x_label.T[2])
            p_x4 = np.mean(x_label.T[3])

            mu_x5 = np.median(x_label.T[4])
            mu_x6 = np.median(x_label.T[5])
            b_x5 = np.mean(np.abs(x_label.T[4] - mu_x5))
            b_x6 = np.mean(np.abs(x_label.T[5] - mu_x6))

            lambda_x7 = 1 / np.mean(x_label.T[6])
            lambda_x8 = 1 / np.mean(x_label.T[7])

            self.gaussian[label] = [mean_x1, mean_x2, var_x1, var_x2]
            self.bernoulli[label] = [p_x3, p_x4]
            self.laplace[label] = [mu_x5, mu_x6, b_x5, b_x6]
            self.exponential[label] = [lambda_x7, lambda_x8]
            self.multinomial[label] = [np.bincount(x_label.T[8].astype(int)) / len(x_label),
                                       np.bincount(x_label.T[9].astype(int)) / len(x_label)]
        """End of your code."""

    def predict(self, x):
        """Start of your code."""
        """
        x : np.array of shape (n,10)

        Calculate the posterior probability using the parameters of the distribution calculated in fit function.
        Take care of underflow errors suitably (Hint: Take log of probabilities)
        Return an np.array() of predictions where predictions[i] is the predicted class for ith data point in X.
        It is implied that prediction[i] is the class that maximizes posterior probability for ith data point in X.
        You can create a separate function for calculating posterior probability and call it here.
        """
        predictions = []
        for label in [0, 1, 2]:
            px1 = -0.5 * np.log(2 * np.pi * self.gaussian[label][2]) - 0.5 * (
                        ((x.T[0] - self.gaussian[label][0]) ** 2) / self.gaussian[label][2])
            px2 = -0.5 * np.log(2 * np.pi * self.gaussian[label][3]) - 0.5 * (
                        ((x.T[1] - self.gaussian[label][1]) ** 2) / self.gaussian[label][3])

            px3 = x.T[2] * np.log(self.bernoulli[label][0]) + (1 - x.T[2]) * np.log(1 - self.bernoulli[label][0])
            px4 = x.T[3] * np.log(self.bernoulli[label][1]) + (1 - x.T[3]) * np.log(1 - self.bernoulli[label][1])

            px5 = -np.log(2 * self.laplace[label][2]) - (
                        (np.absolute(x.T[4] - self.laplace[label][0])) / self.laplace[label][2])
            px6 = -np.log(2 * self.laplace[label][3]) - (
                        (np.absolute(x.T[5] - self.laplace[label][1])) / self.laplace[label][3])

            px7 = np.log(self.exponential[label][0]) - self.exponential[label][0] * x.T[6]
            px8 = np.log(self.exponential[label][1]) - self.exponential[label][1] * x.T[7]

            px9 = np.log(np.array(self.multinomial[label][0])[x.T[8].astype(int)])
            px10 = np.log(np.array(self.multinomial[label][1])[x.T[9].astype(int)])

            p = np.log(self.priors[label]) + px1 + px2 + px3 + px4 + px5 + px6 + px7 + px8 + px9 + px10

            predictions.append(p)

        predictions = np.argmax(np.array(predictions).T, axis=1)

        return predictions

    def getParams(self):
        """
        Return your calculated priors and parameters for all the classes in the form of dictionary that will be used for evaluation
        Please don't change the dictionary names
        Here is what the output would look like:
        priors = {"0":0.2,"1":0.3,"2":0.5}
        gaussian = {"0":[mean_x1,mean_x2,var_x1,var_x2],"1":[mean_x1,mean_x2,var_x1,var_x2],"2":[mean_x1,mean_x2,var_x1,var_x2]}
        bernoulli = {"0":[p_x3,p_x4],"1":[p_x3,p_x4],"2":[p_x3,p_x4]}
        laplace = {"0":[mu_x5,mu_x6,b_x5,b_x6],"1":[mu_x5,mu_x6,b_x5,b_x6],"2":[mu_x5,mu_x6,b_x5,b_x6]}
        exponential = {"0":[lambda_x7,lambda_x8],"1":[lambda_x7,lambda_x8],"2":[lambda_x7,lambda_x8]}
        multinomial = {"0":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"1":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"2":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]]}
        """

        return self.priors, self.gaussian, self.bernoulli, self.laplace, self.exponential, self.multinomial


def save_model(my_model, filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename, "wb")
    pkl.dump(my_model, file)
    file.close()


def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename, "rb")
    my_model = pkl.load(file)
    file.close()
    return my_model


def visualise(data_points, labels):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multiclass f1 score of the predictions.
    For this, we calculate the f1-score for each class

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predicted_labels, actual_labels, class_label):
        """Calculate the multiclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predicted_labels (np.array): The predicted labels.
            actual_labels (np.array): The true labels.
            class_label (int): label value - 0, 1, 2
        Returns:
            float: The precision of the predictions.
        """

        tp, fp = 0, 0
        for predicted_label, actual_label in zip(predicted_labels, actual_labels):
            if predicted_label == class_label and actual_label == class_label:
                tp = tp + 1
            if predicted_label == class_label and actual_label != class_label:
                fp = fp + 1

        return tp / (tp + fp)

    def recall(predicted_labels, actual_labels, class_label):
        """Calculate the multiclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predicted_labels (np.array): The predicted labels.
            actual_labels (np.array): The true labels.
            class_label (int): label value - 0, 1, 2
        Returns:
            float: The recall of the predictions.
        """

        tp, fn = 0, 0
        for predicted_label, actual_label in zip(predicted_labels, actual_labels):
            if actual_label == class_label and predicted_label == class_label:
                tp = tp + 1
            if actual_label == class_label and predicted_label != class_label:
                fn = fn + 1

        return tp / (tp + fn)

    def f1score(predicted_labels, actual_labels, class_label):
        """Calculate the f1 score using its relation with precision and recall.

        Args:
            predicted_labels (np.array): The predicted labels.
            actual_labels (np.array): The true labels.
            class_label (int): label value - 0, 1, 2
        Returns:
            float: The f1 score of the predictions.
        """

        p = precision(predicted_labels, actual_labels, class_label)
        r = recall(predicted_labels, actual_labels, class_label)
        f1 = (2 * p * r) / (p + r)

        return f1

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s


def accuracy(predictions, true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions == true_labels) / predictions.size


if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv', index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv', index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:, :-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    # visualise(validation_datapoints, validation_predictions, "validation_predictions.png")
