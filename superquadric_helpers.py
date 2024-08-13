import numpy as np
from scipy.spatial.transform import Rotation 


def load_superquadric_params(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        shape = list(map(float, lines[0].split()))
        scale = list(map(float, lines[1].split()))
        euler = list(map(float, lines[2].split()))
        translation = list(map(float, lines[3].split()))

    return shape, scale, euler, translation


def save_superquadric_params(save_path, shape, scale, euler, translation):
    # Save the recovered superquadric using the parameters in a txt file where each line is a parameter without brackets
    with open(save_path, 'w') as f:
        f.write('{} {}\n'.format(shape[0], shape[1]))
        f.write('{} {} {}\n'.format(scale[0], scale[1], scale[2]))
        f.write('{} {} {}\n'.format(euler[0], euler[1], euler[2]))
        f.write('{} {} {}\n'.format(translation[0], translation[1], translation[2]))


class Superquadric:

    def __init__(self, epsilon, scale, euler, translation):
        self.epsilon = epsilon
        self.scale = scale
        self.euler = euler
        self.translation = translation

        self.a = self.scale[0]
        self.b = self.scale[1]
        self.c = self.scale[2]
        self.eps_1 = self.epsilon[0]
        self.eps_2 = self.epsilon[1]
        self.eps_1 = max(0.1, self.eps_1)  # otherwise the optimizer has issues
        self.eps_2 = max(0.1, self.eps_2)  # otherwise the optimizer has issues

        self.T = np.eye(4)
        self.T[0:3, 0:3] = Rotation.from_euler('ZYX', euler).as_matrix()
        self.T[0:3, 3] = translation

        self.T_inv = np.linalg.inv(self.T)

    def determine_ellipsoid(self):
        # Determine the ellipsoid that bounds the superquadric
        a = self.a
        b = self.b
        c = self.c

        D = np.diag([1 / a**2, 1 / b**2, 1 / c**2])

        R = self.T[0:3, 0:3]
        t = self.T[0:3, 3]

        # Turn the ellipsoid equation into the form (x - x_0)^T P (x - x_0) <= 1
        self.P = R @ D @ R.T
        self.x_0 = t

    def evaluate_ellipsoid_2(self, point):
        # Transform the point to the superquadric frame
        point = np.append(point, 1)
        point = np.dot(self.T_inv, point)

        x = point[0]
        y = point[1]
        z = point[2]

        # Calculate the ellipsoid function
        f = ((x / self.a)**2 + (y / self.b)**2 + (z / self.c)**2)

        return f
        
    def evaluate_ellipsoid(self, point):
        # Evaluate the ellipsoid equation at a given point
        return (point - self.x_0).T @ self.P @ (point - self.x_0)
    
    def evaluate_ellipsoid_gradient(self, point):
        # Evaluate the gradient of the ellipsoid equation at a given point
        return 2 * (point - self.x_0).T @ self.P

    def robustify(self, radius):
        # Add a small value to the scale to make the superquadric more robust
        self.a_init = self.a.copy()
        self.b_init = self.b.copy()
        self.c_init = self.c.copy()
        
        self.a += radius
        self.b += radius
        self.c += radius

    def evaluate(self, point):
        # Transform the point to the superquadric frame
        point = np.append(point, 1)
        point = np.dot(self.T_inv, point)

        x = point[0]
        y = point[1]
        z = point[2]
        
        # Calculate the superquadric function
        # The order really matters here! The problem is that under the hood the powers 
        # may be calculated in a suboptimal way. This can result in taking the root of a negative number if
        # implemented in the following way: 
        # f = ((x/a)**(2/q) + (y/b)**(2/q))**(q/p) + (z/c)**(2/p) 
        # The second form explicitly first squares the values to avoid negative numbers and then does the roots
        f = (((x/self.a)**2)**(1/self.eps_2) + ((y/self.b)**2)**(1/self.eps_2))**(self.eps_2/self.eps_1) + \
            ((z/self.c)**2)**(1/self.eps_1)

        return f
    
    def transform_point(self, point):
        # Transform the point to the superquadric frame
        point = np.append(point, 1)
        point = np.dot(self.T_inv, point)

        return point
    
    def evaluate_axis_aligned_gradient_ellipsoid(self, point):
        x = point[0]
        y = point[1]
        z = point[2]

        # Calculate the gradient of the ellipsoid function
        dfdx = 2 * x / self.a**2
        dfdy = 2 * y / self.b**2
        dfdz = 2 * z / self.c**2
        
        return np.array([dfdx, dfdy, dfdz])

    def evaluate_axis_aligned_gradient(self, point):
        x = point[0]
        y = point[1]
        z = point[2]

        # Calculate the gradient of the superquadric function
        common_term = (((x/self.a)**2)**(1/self.eps_2) + ((y/self.b)**2)**(1/self.eps_2))**(self.eps_2/self.eps_1 - 1)

        dfdx = (2/(self.eps_1*x))*((x/self.a)**2)**(1/self.eps_2)*common_term
        dfdy = (2/(self.eps_1*y))*((y/self.b)**2)**(1/self.eps_2)*common_term
        dfdz = (2/(self.eps_1*z))*((z/self.c)**2)**(1/self.eps_1)
        
        return np.array([dfdx, dfdy, dfdz])
    
    def evaluate_gradient(self, point):
        # Transform the point to the superquadric frame
        point = np.append(point, 1)
        point = np.dot(self.T_inv, point)

        x = point[0]
        y = point[1]
        z = point[2]

        # Calculate the gradient of the superquadric function
        common_term = (((x/self.a)**2)**(1/self.eps_2) + ((y/self.b)**2)**(1/self.eps_2))**(self.eps_2/self.eps_1 - 1)

        dfdx = (2/(self.eps_1*x))*((x/self.a)**2)**(1/self.eps_2)*common_term
        dfdy = (2/(self.eps_1*y))*((y/self.b)**2)**(1/self.eps_2)*common_term
        dfdz = (2/(self.eps_1*z))*((z/self.c)**2)**(1/self.eps_1)
        
        return np.array([dfdx, dfdy, dfdz])
    
    def numerical_gradient(self, point, evaluate, epsilon=1e-6):
        # Calculate the gradient numerically
        grad = np.zeros(3)
        for i in range(3):
            point_plus = point.copy()
            point_plus[i] += epsilon
            point_minus = point.copy()
            point_minus[i] -= epsilon

            grad[i] = (evaluate(point_plus) - evaluate(point_minus)) / (2 * epsilon)

        return grad
    