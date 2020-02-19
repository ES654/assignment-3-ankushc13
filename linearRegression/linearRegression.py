import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from autograd import grad
import seaborn as sns
import matplotlib.animation as animation
from scipy import stats 
# Import Autograd modules here
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML, Image
from matplotlib.animation import FFMpegWriter
import calendar
import time

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.Xb=None
        self.yb=None
        self.J_history = None
        self.theta_0=[]
        self.theta_1=[]
        pass
    
    def costFunctionReg(self,X,y,theta,lamda = 10):    
        #Initializations
        m = len(y) 
        J = 0

        #Computations
        h = (X@theta).reshape(len(y),)
        J_reg = (lamda / (2*m))*np.sum(np.square(theta))
        J = float((1./(2*m))*((h - y).transpose()).dot(h - y)) + J_reg;
        if np.isnan(J):
            return(np.inf)
        return(J) 


    def data_iter(self,batch_size, X, y):
        batch = len(X)
        indices = list(range(batch))
        # The examples are read at random, in no particular order
        random.shuffle(indices)
        for i in range(0, batch, batch_size):
            batch_indices = indices[i: min(i + batch_size, batch)]
            yield X.loc[batch_indices].reset_index(drop=True), y.loc[batch_indices].reset_index(drop=True)




    def fit_non_vectorised(self, X, y, batch_size=1, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        X1=X.copy()
        
        if self.fit_intercept is True:
            X1.insert(0, 'intercept',[1.0]*y.size)
            self.coef_ = [0]*(X1.shape[1])
            cls=list(X1.columns)
            for i in range(1,n_iter+1):
                for X2, y2 in self.data_iter(batch_size, X1, y):
                    Y_pred = X2.dot(pd.Series(self.coef_,index=cls))
                    n = float(len(X2))
                    for j in range(0,len(self.coef_)):
                        D_m = (-2/n)*sum(X2.iloc[:,j]*(y2 - Y_pred))  # Derivative wrt m
                        if lr_type =='constant':
                            self.coef_[j] = self.coef_[j] - lr*D_m  # Update m
                        else:
                            self.coef_[j] = self.coef_[j] - (lr/float(i))*D_m  # Update m
                
        else:
            self.coef_ = [0]*(X1.shape[1])
            cls=list(X1.columns)
            for i in range(1,n_iter+1):
                for X2, y2 in self.data_iter(batch_size, X1, y):
                    Y_pred = X2.dot(pd.Series(self.coef_,index=cls))
                    n = float(len(X2))
                    for j in range(0,len(self.coef_)):
                        D_m = (-2/n)*sum(X2.iloc[:,j]*(y2 - Y_pred))  # Derivative wrt m
                        if lr_type =='constant':
                            self.coef_[j] = self.coef_[j] - lr*D_m  # Update m
                        else:
                            self.coef_[j] = self.coef_[j] - (lr/float(i))*D_m  # Update m
        
        pass

    def fit_vectorised(self, X, y,batch_size=10, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.J_history = np.zeros(n_iter)
        X1=X.copy()
        if self.fit_intercept is True:
            X1.insert(0, 'intercept',[1.0]*y.size)
            cls=list(X1.columns)
            self.coef_ = pd.Series([2]*(X1.shape[1]),index=cls).astype('float64')
            for i in range(1,n_iter+1):
                for X2, y2 in self.data_iter(batch_size, X1, y):
                    n = float(len(X2))
                    if lr_type =='constant':
                        self.coef_  = self.coef_ - lr*(1/n)*(X2.transpose()).dot((X2.dot(self.coef_) - y2))
                    else:
                        self.coef_  = self.coef_ - (lr/float(i))*(1/n)*(X2.transpose()).dot((X2.dot(self.coef_) - y2))
                self.J_history[i-1] = self.costFunctionReg(X1.to_numpy(),y.to_numpy(),self.coef_.to_numpy(),25)

                self.theta_0.append(self.coef_.iat[0])
                self.theta_1.append(self.coef_.iat[1])

        else:
            cls=list(X1.columns)
            self.coef_ = pd.Series([2]*(X1.shape[1]),index=cls)
            for i in range(1,n_iter+1):
                for X2, y2 in self.data_iter(batch_size, X1, y):
                    n = float(len(X2))
                    if lr_type =='constant':
                        self.coef_  = self.coef_ - lr*(1/n)*(X2.transpose()).dot((X2.dot(self.coef_) - y2))
                    else:
                        self.coef_  = self.coef_ - (lr/float(i))*(1/n)*(X2.transpose()).dot((X2.dot(self.coef_) - y2))
                
                self.J_history[i-1] = self.costFunctionReg(X1.to_numpy(),y.to_numpy(),self.coef_.to_numpy(),25)
                self.theta_0.append(self.coef_.iat[0])
                self.theta_1.append(self.coef_.iat[1])
        pass

    def fit_autograd(self, X, y, batch_size=5, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        gradT = grad(self.training_loss)
        X1=X.copy()
        if self.fit_intercept is True:
            X1.insert(0, 'intercept',[1.0]*y.size)
            cls=list(X1.columns)
            self.coef_ = pd.Series([0]*(X1.shape[1]),index=cls)
            for i in range(1,n_iter+1):
                for self.Xb,self.yb in self.data_iter(batch_size, X1, y):
                    n = float(len(self.Xb))
                    if lr_type =='constant':
                        self.coef_  = self.coef_ - gradT((self.coef_.values).astype('float64'))*(lr)
                    else:
                        self.coef_  = self.coef_ - gradT((self.coef_.values).astype('float64'))*(lr/float(i))
        else:
            cls=list(X1.columns)
            self.coef_ = pd.Series([0]*(X1.shape[1]),index=cls)
            for i in range(1,n_iter+1):
                for self.Xb,self.yb in self.data_iter(batch_size, X1, y):
                    n = float(len(self.Xb))
                    if lr_type =='constant':
                        self.coef_  = self.coef_ - gradT((self.coef_.values).astype('float64'))*(lr)
                    else:
                        self.coef_  = self.coef_ - gradT((self.coef_.values).astype('float64'))*(lr/float(i))


        pass

    def training_loss(self,weights):
        Y_pred = np.dot(self.Xb,weights)
        n = float(len(self.Xb))
        loss = (1/n)*np.sum((self.yb.values - Y_pred)**2)
        return loss

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        X1=X.copy()
        if self.fit_intercept is True:
            X1.insert(0, 'intercept',[1.0]*y.size)
            X_t=X1.transpose()
            df  = X_t.dot(X)
            df_inv = pd.DataFrame(np.linalg.pinv(df.values), df.columns, df.index)
            df2=X_t.dot(y)
            self.coef_=df_inv.dot(df2)
        else:
            X_t=X.transpose()
            df  = X_t.dot(X)
            df_inv = pd.DataFrame(np.linalg.pinv(df.values), df.columns, df.index)
            df2=X_t.dot(y)
            self.coef_=df_inv.dot(df2)
        pass

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X1=X.copy()
        if self.fit_intercept is True:
            X1.insert(0, 'intercept',[1.0]*(X1.shape[0]))
            return X1.dot(self.coef_)
        return X1.dot(self.coef_)
        pass

    def plot_surface(self, X, y):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        X=X.to_numpy()

        y = y.to_numpy()
        T1, T2 = np.meshgrid(np.linspace(-5,5,100),np.linspace(-6,3,100))

        zs = np.array(  [self.costFunctionReg(X[:,0:2], y,np.array([t1,t2]).reshape(-1,1),25) 
                            for t1, t2 in zip(np.ravel(T1), np.ravel(T2)) ] ) 
        Z = zs.reshape(T1.shape)


        fig2 = plt.figure(figsize = (7,7))
        ax2 = Axes3D(fig2)

        #Surface plot
        ax2.plot_surface(T1, T2, Z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
        #ax2.plot(theta_0,theta_1,J_history_reg, marker = '*', color = 'r', alpha = .4, label = 'Gradient descent')

        ax2.set_xlabel('theta 1')
        ax2.set_ylabel('theta 2')
        ax2.set_zlabel('error')
        # ax2.set_title('RSS gradient descent: Root at {}'.format(theta_result_reg.ravel()))
        ax2.view_init(45, -45)

        # Create animation
        line, = ax2.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
        point, = ax2.plot([], [], [], '*', color = 'red')
        display_value = ax2.text(2., 2., 27.5, '', transform=ax2.transAxes)
        def init_2():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            display_value.set_text('')

            return line, point, display_value

        def animate_2(i):
            # Animate line
            line.set_data(self.theta_0[:i], self.theta_1[:i])
            line.set_3d_properties(self.J_history[:i])
            
            # Animate points
            point.set_data(self.theta_0[i], self.theta_1[i])
            point.set_3d_properties(self.J_history[i])

            # Animate display value
            display_value.set_text('Error = ' + str(self.J_history[i]))

            return line, point, display_value
        
        ax2.legend(loc = 1)
        

        anim2 = animation.FuncAnimation(fig2, animate_2, init_func=init_2,
                                    frames=len(self.theta_0), interval=120, 
                                    repeat_delay=60, blit=True)
        writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        anim2.save(str(calendar.timegm(time.gmtime()))+"srfc.mp4", writer=writer )     
        # plt.show()
        # HTML(anim2.to_jshtml())


        






    def plot_line_fit(self, X, y):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        fig, ax = plt.subplots()

        X=X.to_numpy()
        y = y.to_numpy()
        ax.scatter(X[:,1], y)
        line, = ax.plot(X[:,1], y)
        value_display = ax.text(0.02, 0.02, '', transform=ax.transAxes)



        def init():  # only required for blitting to give a clean slate.
            line.set_ydata([np.nan] * len(X))
            value_display.set_text('')

            return line,


        def animate(i):
            line.set_ydata(X[:,1]*self.theta_0[i]+self.theta_1[i])  # update the data.
            value_display.set_text('m = ' + str(self.theta_0[i])+'  b = '+ str(self.theta_1[i]))
            return line, value_display


        ani = animation.FuncAnimation(
            fig, animate, init_func=init, frames=len(self.theta_0),interval=2, blit=True, save_count=50)

        writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(str(calendar.timegm(time.gmtime()))+"line.mp4", writer=writer )     
        # plt.show()

        pass

    def plot_contour(self, X, y):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        X=X.to_numpy()

        y = y.to_numpy()
        T1, T2 = np.meshgrid(np.linspace(-5,5,100),np.linspace(-6,3,100))

        zs = np.array(  [self.costFunctionReg(X[:,0:2], y,np.array([t1,t2]).reshape(-1,1),25) 
                            for t1, t2 in zip(np.ravel(T1), np.ravel(T2)) ] ) 
        Z = zs.reshape(T1.shape)
        #Plot the contour
        fig1, ax1 = plt.subplots(figsize = (7,7))
        ax1.contour(T1, T2, Z, 100, cmap = 'jet')


        # Create animation
        line, = ax1.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
        point, = ax1.plot([], [], '*', color = 'red', markersize = 4)
        value_display = ax1.text(0.02, 0.02, '', transform=ax1.transAxes)

        def init_1():
            line.set_data([], [])
            point.set_data([], [])
            value_display.set_text('')

            return line, point, value_display

        def animate_1(i):
            # Animate line
            line.set_data(self.theta_0[:i], self.theta_1[:i])
            
            # Animate points
            point.set_data(self.theta_0[i], self.theta_1[i])

            # Animate value display
            value_display.set_text('Error = ' + str(self.J_history[i]))

            return line, point, value_display

        ax1.legend(loc = 1)

        anim1 = animation.FuncAnimation(fig1, animate_1, init_func=init_1,
                                    frames=len(self.theta_0), interval=100, 
                                    repeat_delay=60, blit=True)
        writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        anim1.save(str(calendar.timegm(time.gmtime()))+"cntr.mp4", writer=writer )     
        #plt.show()
        # HTML(anim1.to_jshtml())
        pass
    

    def print_theta(self,X,y):
        dist = np.linalg.norm(self.coef_)
        return dist

