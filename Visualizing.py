import matplotlib.pyplot as plt
import autograd.numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad, grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

class Vplot():
    def __init__(self, func, xmin, xmax, ymin, ymax, step):
        super().__init__()
        self.f = func
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.step = step
        self.x ,self.y = np.meshgrid(np.arange(xmin, xmax + step, step), np.arange(ymin, ymax + step, step))
        self.z = self.f(self.x, self.y)

    def Surface_Plot_3D(self):
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection='3d', elev=50, azim=-50)

        ax.plot_surface(self.x, self.y, self.z, norm=LogNorm(), rstride=1, cstride=1, 
                        edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')

        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))

        plt.show()

    def gradient_field_Plot_2D(self):
        dz_dx = elementwise_grad(self.f, argnum=0)(self.x, self.y)
        dz_dy = elementwise_grad(self.f, argnum=1)(self.x, self.y)

        fig, ax = plt.subplots(figsize=(10, 6))

        # ax.contour(self.x, self.y, self.z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        ax.contour(self.x, self.y, self.z, cmap=plt.cm.jet)

        ax.quiver(self.x, self.y, self.x - dz_dx, self.y - dz_dy, alpha=.5)
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))

        plt.show()
    
    def Gradient_Descent(self,Grad1,Grad2,start_point,gamma = 0.00125, epsilon=0.0001, nMax = 10000 ):
        # x,y : start point
        i = 0
        iter_x, iter_y, iter_count = np.empty(0),np.empty(0), np.empty(0)
        error = 10
        X = start_point
        x,y = X[0], X[1] 
        #Looping as long as error is greater than epsilon
        while np.linalg.norm(error) > epsilon and i < nMax:
            i +=1
            iter_x = np.append(iter_x,x)
            iter_y = np.append(iter_y,y)
            iter_count = np.append(iter_count ,i)   
            #print(X) 
            
            X_prev = X
            Grad = np.array([Grad1(x,y),Grad2(x,y)])
            X = X - gamma * Grad
            error = X - X_prev
            x,y = X[0], X[1]
        # print(X)
        # print(iter_x, iter_y)
        return X, iter_x, iter_y, iter_count

    def SGD_plot(self,start_point,gamma = 0.00125, epsilon=0.0001, nMax = 10000):
        Grad1 = grad(self.f, argnum=0)
        Grad2 = grad(self.f, argnum=1)
        X,iter_x, iter_y, iter_count = self.Gradient_Descent(Grad1,Grad2,start_point,gamma, epsilon, nMax)
        #Angles needed for quiver plot
        anglesx = iter_x[1:] - iter_x[:-1]
        anglesy = iter_y[1:] - iter_y[:-1]

        fig = plt.figure(figsize = (12,5))

        #Surface plot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, norm=LogNorm(), rstride=5, cstride=5, 
                        edgecolor='none', alpha=.4, cmap=plt.cm.jet)
        ax.plot(iter_x,iter_y, self.f(iter_x,iter_y),color = 'r', marker = '*', alpha = .4)
        ax.plot(*X.reshape(-1,1), self.f(*X.reshape(-1,1)), 'r*', markersize=18)
        ax.view_init(45, 280)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Contour plot
        ax = fig.add_subplot(1, 2, 2)
        ax.contour(self.x, self.y, self.z, 50, cmap=plt.cm.jet)
        #Plotting the iterations and intermediate values
        ax.scatter(iter_x,iter_y,color = 'r', marker = '*')
        ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'k')
        ax.plot(*X.reshape(-1,1), 'r*', markersize=18)
        ax.set_title('Gradient Descent with {} iterations'.format(len(iter_count)))
        plt.show()

    def make_minimize_cb(self, path=[]): 
        def minimize_cb(xk):
            path.append(np.copy(xk))
        return minimize_cb

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def Scipy_optimization(self,method_name,start_point):
        func = value_and_grad(lambda args: self.f(*args))
        x0 = start_point
        path_ = [x0]
        res = minimize(func, x0=x0, method=method_name,jac=True, tol=1e-20, callback=self.make_minimize_cb(path_))
        path = np.array(path_).T

        fig = plt.figure(figsize = (12,5))
        #Surface plot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, norm=LogNorm(), rstride=5, cstride=5, 
                        edgecolor='none', alpha=.4, cmap=plt.cm.jet) 
        ax.plot(path[0],path[1], self.f(path[0],path[1]),color = 'r', marker = '*', alpha = .4) 
        # ax.quiver(path[0,:-1], path[1,:-1], self.f(*path[::,:-1]), path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], self.f(*(path[::,1:]-path[::,:-1])), 
        #         color='k', alpha = .3)
        ax.plot(*res['x'].reshape(-1,1), self.f(*res['x'].reshape(-1,1)), 'r*', markersize=18)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        
        # Contour plot
        ax = fig.add_subplot(1, 2, 2)
        ax.contour(self.x, self.y, self.z, 50, cmap=plt.cm.jet)
        ax.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
        ax.plot(*res['x'].reshape(-1,1), 'r*', markersize=18)
        plt.title(method_name)
        plt.show()

    def Contour_plot_video(self, method_name, start_point):
        def animate(i):
            line.set_data(*path[::,:i])
            point.set_data(*path[::,i-1:i])
            return line, point
        func = value_and_grad(lambda args: self.f(*args))
        x0 = start_point
        path_ = [x0]
        res = minimize(func, x0=x0, method=method_name,jac=True, tol=1e-20, callback=self.make_minimize_cb(path_))
        path = np.array(path_).T

        fig, ax = plt.subplots(figsize=(12,5))

        ax.contour(self.x, self.y, self.z, 50, cmap=plt.cm.jet)
        ax.plot(*res['x'].reshape(-1,1), 'r*', markersize=18)
        line, = ax.plot([], [], 'b', label=method_name, lw=2)
        point, = ax.plot([], [], 'bo')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.legend(loc='upper left')
        anim = animation.FuncAnimation(fig, animate, init_func=self.init,
                               frames=path.shape[1], interval=60, 
                               repeat_delay=5, blit=True)
        HTML(anim.to_html5_video())
