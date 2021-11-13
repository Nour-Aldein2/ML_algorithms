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

#################################################
########### Show the training data
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(x_train[y_train == 1, -2], x_train[y_train == 1, -1], 'bx', linewidth=2)
plt.plot(x_train[y_train == 0, -2], x_train[y_train == 0, -1], 'go', linewidth=2)
# plt.show()
#################################################







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
util.plot(x_train, y_train, theta=log_reg.theta)
print("Theta is: ", log_reg.theta)
print("The accuracy on training set is: ", np.mean(log_reg.predict(x_train) == y_train))

util.plot(x_valid, y_valid, log_reg.theta)
print("The accuracy on validation set is: ", np.mean(log_reg.predict(x_valid) == y_valid))


plt.show()
