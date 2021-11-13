# Note: this file is a combaniation of the GDA and the logistic regression problems


import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)       # change it to false to get rid of the intercept term (x_0)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    return x_train, y_train, x_valid, y_valid

    # *** START CODE HERE ***
# Specify the locations as discribed in the problem above.
train = 'C:\\Users\\nour_\\OneDrive\\QI\\Machine_Learning\\cs229-2018-autumn-main\\problem-sets\\PS1\\data\\ds1_train.csv'
eval = 'C:\\Users\\nour_\\OneDrive\\QI\\Machine_Learning\\cs229-2018-autumn-main\\problem-sets\\PS1\\data\\ds1_valid.csv'
pred = 'C:\\Users\\nour_\\OneDrive\\QI\\Machine_Learning\\cs229-2018-autumn-main\\problem-sets\\PS1\\predictions_p1'

# Call the function and store the training data
x_train, y_train, x_valid, y_valid = main(train, eval, pred)

m = len(y_train)        #length of training data set
n = len(x_train[0])
    # *** END CODE HERE ***

""" Note:
theta is n-by-1 vector
x is m-by-n matrix
h is m-by-1 vector
"""


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def h(theta,x):
            return 1 / ( 1 + np.exp(-np.dot(x,theta)) )

        def grad_J(theta,x,y):
            # Gradiant of J(theta)
            return np.dot(x.transpose(),(h(theta,x)-y)) / m

        def Hessian(theta,x):
            # h = np.reshape(h(theta,x), (-1,1))           # We need to reshape h so we can compute the Hessian
            return np.dot(x.transpose(),np.reshape(h(theta,x), (-1,1)) * (1 - np.reshape(h(theta,x), (-1,1))) * x)

        def next_theta(theta,x,y):
            H_inv = np.linalg.inv(Hessian(theta,x))    #compute the inverse of a matrix
            return theta - np.dot(H_inv,grad_J(theta,x,y))
        # Initialize theta
        if self.theta is None:
            self.theta = np.zeros(n)

        # Update theta using Newton's Method
        old_theta = self.theta
        new_theta = next_theta(self.theta, x, y)
        while np.linalg.norm(new_theta - old_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(old_theta, x, y)

        self.theta = new_theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        :param x: Inputs of shape (m, n).
        :return:  Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return x @ self.theta >= 0

        # *** END CODE HERE ***

log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)




def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset

    x_train, y_train = util.load_dataset(train_path)       # change it to false to get rid of the intercept term (x_0)
    x_valid, y_valid = util.load_dataset(eval_path)
    return x_train, y_train, x_valid, y_valid

    # *** START CODE HERE ***
# Specify the locations as discribed in the problem above.
train = 'C:\\Users\\nour_\\OneDrive\\QI\\Machine_Learning\\cs229-2018-autumn-main\\problem-sets\\PS1\\data\\ds1_train.csv'
eval = 'C:\\Users\\nour_\\OneDrive\\QI\\Machine_Learning\\cs229-2018-autumn-main\\problem-sets\\PS1\\data\\ds1_valid.csv'
pred = 'C:\\Users\\nour_\\OneDrive\\QI\\Machine_Learning\\cs229-2018-autumn-main\\problem-sets\\PS1\\predictions_p1'

# Call the function and store the training data
x_train, y_train, x_valid, y_valid = main(train, eval, pred)
    # *** END CODE HERE ***



##############################################################
#################### Plotting the data #######################
# plt.plot(x_train[y_train == 1, -2], x_train[y_train == 1, -1], 'bx', linewidth=2)
# plt.plot(x_train[y_train == 0, -2], x_train[y_train == 0, -1], 'go', linewidth=2)
# plt.show()
##############################################################
##############################################################

m, n = x_train.shape
x,y = x_train,y_train

class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        def phi(y):
            s = 0
            for i in range (len (y) ):
                s += y[i]          # you can use conditional statement to make this line.
            return s/len(y)

        def mu_0(x,y):
            numerator = denominator = 0
            for i in range ( len (y) ):
                if y[i] == 1:
                    continue
                else:
                    indicator = 1
                numerator += indicator * x[i]
                denominator += indicator
            return (numerator/denominator)     # .reshape(3,1) in order to make it a vector (so we can get sigma)

        def mu_1(x,y):
            numerator = denominator = 0
            for i in range ( len (y) ):
                if y[i] == 0:
                    continue
                else:
                    indicator = 1
                numerator += indicator * x[i]
                denominator += indicator
            return (numerator/denominator)    # .reshape(3,1) in order to make it a vector (so we can get sigma)


        def sigma(x,y):
            s = 0
            # To make this function faster try to store mu_1 and mu_0 in variables
            # and substitute the values of them instead of calling them again and again.
            for i in range( len(y) ):
                y_i = np.reshape(y[i], (1,-1))

                if y[i] == 0:
                    x_centered = x[i] - (1-y_i)*mu_0(x,y)
                    s += np.dot (x_centered.T,x_centered)
                else:
                    x_centered = x[i] - y_i*mu_1(x,y)
                    s += np.dot (x_centered.T,x_centered)
            return s/len(y)

        def theta(x,y):
            return np.dot ( np.linalg.inv( (sigma(x,y)) ), (mu_1(x,y)-mu_0(x,y)) ).reshape(n,)

        def theta_0(x,y):
            sigma_inv = np.linalg.inv(sigma(x,y))
            mu_plus = mu_0(x,y) + mu_1(x,y)
            mu_minus = mu_0(x,y) - mu_1(x,y)
            return  np.dot( np.dot(mu_plus.T , sigma_inv) , mu_minus)/2  + np.log( (1-phi(y))/phi(y) )

        self.theta1 = theta(x,y)
        self.theta0 = theta_0(x,y)

        self.theta = np.insert(theta(x,y), 0, theta_0(x,y))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # Add x_0 = 1 convention to make predictions using theta^T x >= 0
        return util.add_intercept(x) @ self.theta >= 0
        # *** END CODE HERE





# Plotting the data set
# plt.plot(x_train[y_train == 1, -2], x_train[y_train == 1, -1], 'bx')
# plt.plot(x_train[y_train == 0, -2], x_train[y_train == 0, -1], 'm.')

def plot(x, y, theta_1, legend_1=None, theta_2=None, legend_2=None, title=None, correction=1.0):
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta_1^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta_1[0] / theta_1[2] * correction + theta_1[1] / theta_1[2] * x1)
    plt.plot(x1, x2, c='red', label=legend_1, linewidth=2)

    # Plot decision boundary (found by solving for theta_2^T x = 0)
    if theta_2 is not None:
        x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
        x2 = -(theta_2[0] / theta_1[2] * correction + theta_2[1] / theta_2[2] * x1)
        plt.plot(x1, x2, c='black', label=legend_2, linewidth=2)

    # Add labels, legend and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    if legend_1 is not None or legend_2 is not None:
        plt.legend(loc="upper left")
    if title is not None:
        plt.suptitle(title, fontsize=12)





gda = GDA()
gda.fit(x_train, y_train)

plot(x_valid, y_valid, theta_1=log_reg.theta, legend_1='logistic regression', theta_2=gda.theta, legend_2='GDA', title='Validation Set 1')
plt.show()
