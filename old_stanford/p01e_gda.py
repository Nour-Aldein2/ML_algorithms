import numpy as np
import util

from linear_model import LinearModel

import matplotlib.pyplot as plt

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
print(m,n)
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



gda = GDA()
gda.fit(x_train, y_train)
gda.fit(x_valid, y_valid)

util.plot(x_valid, y_valid, theta=gda.theta)

plt.show()
