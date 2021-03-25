"""
## =========================================================================== ## 
MIT License
Copyright (c) 2020 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ## 
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: manipulator.py
## =========================================================================== ## 
"""

# Numpy (Array computing Lib.) [pip3 install numpy]
import numpy as np
# Matplotlib (Visualization Lib.) [pip3 install matplotlib]
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DH_parameters(object):
    # << DH (Denavit-Hartenberg) parameters structure >> #
    def __init__(self, theta, a, d, alpha):
        # Angle about previous z, from old x to new x
        # Unit [radian]
        self.theta = theta
        # Length of the common normal. Assuming a revolute joint, this is the radius about previous z
        # Unit [metres]
        self.a = a
        # Offset along previous z to the common normal 
        # Unit [metres]
        self.d = d
        # Angle about common normal, from old z axis to new z axis
        # Unit [radian]
        self.alpha = alpha

class Control(object):
    def __init__(self, robot_name, rDH_param, ax_wr):
        # << PUBLIC >> #
        # Robot DH (Denavit-Hartenberg) parameters 
        self.rDH_param  = rDH_param
        # Robot Name -> Does not affect functionality (only for user)
        self.robot_name = robot_name
        # Axis working range (Transform to radians)
        self.ax_wr = [x * (np.pi/180) for sublist in ax_wr for x in sublist]
        # Translation Part -> p(x, y)
        self.p = np.zeros(2)
        # Joints Rotation -> theta(theta_1, theta_2)
        self.theta = np.zeros(2)
        # Jacobian Matrix (For the Jacobian Inverse Kinematics Calculation)
        self.jacobian_matrix = np.matrix(np.identity(2))
        # Control of the Visualization
        self.plt   = plt
        self.figure = self.plt.figure(num=None, figsize=(25, 17.5), dpi=80, facecolor='w', edgecolor='k')
        # Resulting trajectory
        self.trajectory = [[], [], []]
        # Visible -> Robot Base Recantgle
        self.plt.gca().add_patch(
            patches.Rectangle((-0.05, -0.05), 0.1, 0.1, label=r'Robot Base', facecolor=[0, 0, 0, 0.25]
            )
        )
        # Visible -> Joint 1
        self.plt.plot(0.0, 0.0, 
            label=r'Joint 1: $\theta_1 ('+ str(self.ax_wr[0] * (180/np.pi)) +','+ str(self.ax_wr[1] * (180/np.pi)) +')$', 
            marker = 'o', ms = 25, mfc = [0,0,0], markeredgecolor = [0,0,0], mew = 5
        )
        # << PRIVATE >> #
        # Transformation matrix for FK calculation [4x4]
        self.__Tn_theta   = np.zeros((4, 4), dtype=np.float)
        # Auxiliary variables -> Target (Translation, Joint/Rotation) for calculation FK/IK
        self.__p_target     = None
        self.__theta_target = np.zeros(2)
        # Rounding index of calculation (accuracy)
        self.__rounding_index = 10
        # Display information about the results of DH Parameters.
        self.__display_rDHp()
        # Visible -> Arm 1: Base <-> Joint 1
        __line1, = self.plt.plot([],[],
            'k-',linewidth=10
        )
        # Visible -> Arm 2: Joint 2 <-> End-Effector
        __line2, = self.plt.plot([],[],
            'k-',linewidth=10
        )     
        # Visible -> Joint 2
        __line3,  = self.plt.plot([],[],
            label=r'Joint 2: $\theta_2 ('+ str(self.ax_wr[2] * (180/np.pi)) +','+ str(self.ax_wr[3] * (180/np.pi)) +')$', marker = 'o', ms = 15, mfc = [0.7, 0.0, 1, 1], markeredgecolor = [0,0,0], mew = 5
        )
        # Visible -> End-Effector
        __line4,  = self.plt.plot([],[],
            label=r'End-Effector Position: $EE_{(x, y)}$', marker = 'o', ms = 15, mfc = [0,0.75,1, 1], markeredgecolor = [0,0,0], mew = 5
        )
        # Arrays of individual objects (for animation)
        self.__line = [__line1, __line2, __line3, __line4]
        # Animation trajectory Matrix (the resulting matrix will contain each of the positions for the animation objects)
        self.__animation_dMat = np.zeros((1, 4), dtype=np.float)

    def __fast_calc_fk(self):
        """
        Description: 
            Fast calculation of forward kinematics (in this case it is recommended to use).
        """

        self.p[0] = round(self.rDH_param.a[0]*np.cos(self.rDH_param.theta[0]) + self.rDH_param.a[1]*np.cos(self.rDH_param.theta[0] + self.rDH_param.theta[1]), self.__rounding_index)
        self.p[1] = round(self.rDH_param.a[0]*np.sin(self.rDH_param.theta[0]) + self.rDH_param.a[1]*np.sin(self.rDH_param.theta[0] + self.rDH_param.theta[1]), self.__rounding_index)

    def __dh_calc_fk(self, index):
        """
        Description: 
            Slower calculation of Forward Kinematics using the Denavit-Hartenberg parameter table.
        Args:
            (1) index [INT]: Index of episode (Number of episodes is depends on number of joints)
        Returns:
            (2) Ai_aux [Float Matrix 4x4]: Transformation Matrix in the current episode

        Examples:
            self.forward_kinematics(0, [0.0, 45.0])
        """

        # Reset/Initialize matrix
        Ai_aux = np.matrix(np.identity(4), copy=False)
        
        # << Calulation First Row >>
        # Rotational Part
        Ai_aux[0, 0] = np.cos(self.rDH_param.theta[index])
        Ai_aux[0, 1] = (-1)*(np.sin(self.rDH_param.theta[index]))*np.cos(self.rDH_param.alpha[index])
        Ai_aux[0, 2] = np.sin(self.rDH_param.theta[index])*np.sin(self.rDH_param.alpha[index])
        # Translation Part
        Ai_aux[0, 3] = self.rDH_param.a[index]*np.cos(self.rDH_param.theta[index])

        # << Calulation Second Row >>
        # Rotational Part
        Ai_aux[1, 0] = np.sin(self.rDH_param.theta[index])
        Ai_aux[1, 1] = np.cos(self.rDH_param.theta[index])*np.cos(self.rDH_param.alpha[index])
        Ai_aux[1, 2] = (-1)*(np.cos(self.rDH_param.theta[index]))*(np.sin(self.rDH_param.alpha[index]))
        # Translation Part
        Ai_aux[1, 3] = self.rDH_param.a[index]*np.sin(self.rDH_param.theta[index])

        # << Calulation Third Row >>
        # Rotational Part
        Ai_aux[2, 0] = 0
        Ai_aux[2, 1] = np.sin(self.rDH_param.alpha[index])
        Ai_aux[2, 2] = np.cos(self.rDH_param.alpha[index])
        # Translation Part
        Ai_aux[2, 3] = self.rDH_param.d[index]

        # << Set Fourth Row >>
        # Rotational Part
        Ai_aux[3, 0] = 0
        Ai_aux[3, 1] = 0
        Ai_aux[3, 2] = 0
        # Translation Part
        Ai_aux[3, 3] = 1

        return Ai_aux

    def __separete_translation_part(self):
        """
        Description: 
            Separation translation part from the resulting transformation matrix.
        """

        for i in range(len(self.p)):
            self.p[i] = round(self.__Tn_theta[i, 3], self.__rounding_index)

    def forward_kinematics(self, calc_type, theta, degree_repr):
        """
        Description:
            Forward kinematics refers to the use of the kinematic equations of a robot to compute 
            the position of the end-effector from specified values for the joint parameters.
            Joint Angles (Theta_1, Theta_2) <-> Position of End-Effector (x, y)
        Args:
            (1) calc_type [INT]: Select the type of calculation (0: DH Table, 1: Fast).
            (2) theta [Float Array]: Joint angle of target in degrees.
            (3) degree_repr [BOOL]: Representation of the input joint angle (Degree).

        Examples:
            self.forward_kinematics(0, [0.0, 45.0])
        """
        self.__theta_target = np.zeros(2)
        self.__theta_target[0] = theta[0]
        self.__theta_target[1] = theta[1]

        if degree_repr == True:
            self.rDH_param.theta = [x * (np.pi/180) for x in self.__theta_target]
        else:
            self.rDH_param.theta = self.__theta_target

        if calc_type == 0:
            for i in range(len(self.rDH_param.theta)):
                self.__Tn_theta =  self.__Tn_theta * self.__dh_calc_fk(i)

            self.__separete_translation_part()
        elif calc_type == 1:
            self.__fast_calc_fk()

        # After completing the calculation, reset the transformation matrix.
        self.__Tn_theta = np.matrix(np.identity(4))

    def inverse_kinematics(self, p, cfg):
        """
        Description:
            Inverse kinematics is the mathematical process of calculating the variable 
            joint parameters needed to place the end of a kinematic chain.
            Position of End-Effector (x, y) <-> Joint Angles (Theta_1, Theta_2)
        Args:
            (1) p [Float Array]: Position (x, y) of the target in meters.
            (2) cfg [INT]: Robot configuration (IK Multiple Solutions).

        Examples:
            self.inverse_kinematics([0.45, 0.10], 0)
        """

        theta_aux     = np.zeros(2)
        self.__p_target = np.zeros(2)
        self.__p_target[0] = p[0]
        self.__p_target[1] = p[1]

        # Cosine Theorem [Beta]: eq (1)
        cosT_beta_numerator   = ((self.rDH_param.a[0]**2) + (self.__p_target[0]**2 + self.__p_target[1]**2) - (self.rDH_param.a[1]**2))
        cosT_beta_denumerator = (2*self.rDH_param.a[0]*np.sqrt(self.__p_target[0]**2 + self.__p_target[1]**2))
        
        # Calculation angle of Theta 1,2 (Inverse trigonometric functions):
        # Rule 1: The range of the argument “x” for arccos function is limited from -1 to 1.
        # −1 ≤ x ≤ 1
        # Rule 2: Output of arccos is limited from 0 to π (radian).
        # 0 ≤ y ≤ π

        # Calculation angle of Theta 1
        if cosT_beta_numerator/cosT_beta_denumerator > 1:
            theta_aux[0] = np.arctan2(self.__p_target[1], self.__p_target[0]) 
            print('[INFO] Theta 1 Error: ', self.__p_target[0], self.__p_target[1])
        elif cosT_beta_numerator/cosT_beta_denumerator < -1:
            theta_aux[0] = np.arctan2(self.__p_target[1], self.__p_target[0]) - np.pi 
            print('[INFO] Theta 1 Error: ', self.__p_target[0], self.__p_target[1]) 
        else:
            if cfg == 0:
                theta_aux[0] = np.arctan2(self.__p_target[1], self.__p_target[0]) - np.arccos(cosT_beta_numerator/cosT_beta_denumerator)
            elif cfg == 1:
                theta_aux[0] = np.arctan2(self.__p_target[1], self.__p_target[0]) + np.arccos(cosT_beta_numerator/cosT_beta_denumerator)
                
        # Cosine Theorem [Alha]: eq (2)
        cosT_alpha_numerator   = (self.rDH_param.a[0]**2) + (self.rDH_param.a[1]**2) - (self.__p_target[0]**2 + self.__p_target[1]**2)
        cosT_alpha_denumerator = (2*(self.rDH_param.a[0]*self.rDH_param.a[1]))

        # Calculation angle of Theta 2
        if cosT_alpha_numerator/cosT_alpha_denumerator > 1:
            theta_aux[1] = np.pi
            print('[INFO] Theta 2 Error: ', self.__p_target[0], self.__p_target[1])
        elif cosT_alpha_numerator/cosT_alpha_denumerator < -1:
            theta_aux[1] = 0.0
            print('[INFO] Theta 2 Error: ', self.__p_target[0], self.__p_target[1])
        else:
            if cfg == 0:
                theta_aux[1] = np.pi - np.arccos(cosT_alpha_numerator/cosT_alpha_denumerator)
            elif cfg == 1:
                theta_aux[1] = np.arccos(cosT_alpha_numerator/cosT_alpha_denumerator) - np.pi

        self.theta = theta_aux

        # Calculate the forward kinematics from the results of the inverse kinematics.
        self.forward_kinematics(1, self.theta, False)

    def inverse_kinematics_jacobian(self, p_target, theta, accuracy, num_of_iter):
        """
        Description:
            The Jacobian matrix method is an incremental method of inverse kinematics 
            (the motion required to move a limb to a certain position may be performed over several frames). 
        Args:
            (1) p_target [Float Array]: Position (x, y) of the target in meters.
            (2) theta [Float Array]: Joint angle of target in radians.
            (3) accuracy [Float]: Accuracy of inverse kinematics calculation.
            (4) num_of_iter [INT]: Number of iterations of the calculation.

        Examples:
            self.inverse_kinematics_jacobian([0.35, 0.15], [0.0, 0.0], 0.0001, 10000)
        """

        # Add a small value to each part of theta (Problem with 0.0 value)
        theta_actual = [theta[0] + 0.01, theta[0] + 0.01]

        # Calculation of FK to find the position p (x, y) for actual theta
        self.forward_kinematics(0, [theta_actual[0], theta_actual[1]], False)

        self.__p_target = np.zeros(2)
        self.__p_target[0] = p_target[0]
        self.__p_target[1] = p_target[1]

        p_start = np.zeros(2)
        p_start = [self.p[0], self.p[1]]

        # Check and change the math sign (positive/negative value of the position {initial, target})
        if p_start[0] > self.__p_target[0]:
            theta_actual[0] *= (-1)
        else:
            theta_actual[0] *= (1)

        if p_start[1] > self.__p_target[1]:
            theta_actual[1] *= (-1)
        else:
            theta_actual[1] *= (1)
        
        # The resulting trajectory of the calculation IK{Jacobian}
        x_traj = []
        y_traj = []

        # Variable for the calculation error (theta new is out of range)
        calc_err = False

        for i in range(num_of_iter):
            self.forward_kinematics(0, [theta_actual[0], theta_actual[1]], False)

            x_traj.append(self.p[0])
            y_traj.append(self.p[1])

            x_actual = abs(self.__p_target[0] - self.p[0])
            y_actual = abs(self.__p_target[1] - self.p[1])
            
            if x_actual < accuracy and y_actual < accuracy:
                print('[INFO] Result found in iteration no. ', i)
                print('[INFO] Target Position (End-Effector):')
                print('[INFO] p_t  = [x: %f, y: %f]' % (self.__p_target[0], self.__p_target[1]))
                print('[INFO] Actual Position (End-Effector):')
                print('[INFO] p_ee = [x: %f, y: %f]' % (self.p[0], self.p[1]))
                print('[INFO] IK Jacobian (Accuracy Error):')
                print('[INFO] Euclidean Distance: %f' % np.linalg.norm(self.p - self.__p_target))
                break

            # Get Jacobian Matrix from actual theta
            self.__calc_jacobian_matrix([theta_actual[0], theta_actual[1]])

            # Calculation of Jac. Matrix determinant (for finding singularities)
            jacobian_Det = np.linalg.det(self.jacobian_matrix)

            # Singular Jacobian -> det(J) = 0 
            if jacobian_Det != 0:
                # Calculation of Inverse Jacobian
                jacobian_Inv = np.linalg.inv(self.jacobian_matrix)

                # Derivation of the theta -> dTheta = J^(-1)*v, where v is derivation of the position (x_actual, y_actual)
                d_theta = np.dot(jacobian_Inv, np.array([[x_actual], [y_actual]]))

                if calc_err == True:
                    theta_actual = [theta[0] + 0.01, theta[0] + 0.01]

                    x_traj.clear()
                    y_traj.clear()

                    x_traj.append(p_start[0])
                    y_traj.append(p_start[1])

                # Add theta derivation to the actual theta value (new position of the trajectory)
                theta_actual[0] = theta_actual[0] + d_theta[0,0]*0.1
                theta_actual[1] = theta_actual[1] + d_theta[1,0]*0.1

                if abs(d_theta[0,0]*0.1) > self.ax_wr[0][1] * (np.pi/180) or abs(d_theta[1,0]*0.1) > self.ax_wr[1][1] * (np.pi/180):
                    # Theta new is out of range -> reset the calculation in the next step
                    calc_err = True
                else:
                    calc_err = False
            else:
                print('[INFO] Singular Jacobian (Problem):')
                print('[INFO] Jacobian Determinant Result: %f' % jacobian_Det)
                break
            
        self.theta = theta_actual

    def __calc_jacobian_matrix(self, theta):
        """
        Description:
            Creating a Jacobian matrix for the actual value of theta.

        Args:
            (1) theta [Float Array]: Joint angle of target in radians.
        """

        self.jacobian_matrix[0, 0] = round(((-1)*self.rDH_param.a[0]*np.sin(theta[0])) + ((-1)*self.rDH_param.a[1]*np.sin(theta[0] + theta[1])), self.__rounding_index)
        self.jacobian_matrix[0, 1] = round((-1)*self.rDH_param.a[1]*np.sin(theta[0] + theta[1]), self.__rounding_index)
        self.jacobian_matrix[1, 0] = round((self.rDH_param.a[0]*np.cos(theta[0])) + (self.rDH_param.a[1]*np.cos(theta[0] + theta[1])), self.__rounding_index)
        self.jacobian_matrix[1, 1] = round((self.rDH_param.a[1]*np.cos(theta[0] + theta[1])), self.__rounding_index)

    def generate_trajectory(self, trajectory_str):
        """
        Description: 
            The function shows the trajectory generation for linear and joint interpolation. A linear Bézier curve is used to generate linear interpolation.

        Args:
            (1) trajectory_str [Structure Type Array]: Structure of the trajectory points.

        Return:
            (1 - 2) parameter{1}, parameter{2} [Float Array]: Results of trajectory values (x, y).
            (3) parameter{3} [INT]: Results of trajectory values (Inverse Kinematics config).
        
        Example (Input Structure):
            Linear Interpolation: {'interpolation': 'linear', 'start_p': [0.50, 0.0], 'target_p': [x[0], y[0]], 'step': 100, 'cfg': 1}
            Joint Interpolation: {'interpolation': 'joint', 'start_p': [0.50, 0.0], 'target_p': [x[0], y[0]], 'step': 100, 'cfg': 1}
        """

        cfg = [trajectory_str['cfg']] * trajectory_str['step']

        self.__p_target = [trajectory_str['target_p'][0], trajectory_str['target_p'][1]]
        self.__theta_target = self.inverse_kinematics(self.__p_target, trajectory_str['cfg'])

        if trajectory_str['interpolation'] == 'linear':
            """
            Another approach:
            
            x = np.linspace(trajectory_str['start_p'][0], trajectory_str['target_p'][0], trajectory_str['step'])
            y = np.linspace(trajectory_str['start_p'][1], trajectory_str['target_p'][1], trajectory_str['step'])
            """

            time = np.linspace(0.0, 1.0, trajectory_str['step'])

            # Linear Bezier Curve
            # p(t) = (1 - t)*p_{0} + t*p_{1}, t ∈ [0, 1]
            x = (1 - time) * trajectory_str['start_p'][0] + time * trajectory_str['target_p'][0]
            y = (1 - time) * trajectory_str['start_p'][1] + time * trajectory_str['target_p'][1]
    
        elif trajectory_str['interpolation'] == 'joint':
            x   = []
            y   = []
            
            self.inverse_kinematics(trajectory_str['start_p'], trajectory_str['cfg'])
            start_theta  = self.rDH_param.theta

            self.inverse_kinematics(trajectory_str['target_p'], trajectory_str['cfg'])
            target_theta = self.rDH_param.theta

            start_theta_dt  = np.linspace(start_theta[0], target_theta[0], trajectory_str['step'])
            target_theta_dt = np.linspace(start_theta[1], target_theta[1], trajectory_str['step'])

            for i in range(len(start_theta_dt)):
                self.forward_kinematics(1, [start_theta_dt[i], target_theta_dt[i]], False)
                x.append(self.p[0])
                y.append(self.p[1])
            
        # Display the trajectory results with start and end point  
        self.plt.plot(x[0], y[0], marker = 'o', ms = 15, mfc = [1,1,0], markeredgecolor = [0,0,0], mew = 5)
        self.plt.plot(x[len(x) - 1], y[len(y) - 1], marker = 'o', ms = 15, mfc = [1,1,0], markeredgecolor = [0,0,0], mew = 5)
        self.plt.plot(x, y, 'g--', linewidth=3.0)

        return [x, y, cfg]

    def smooth_trajectory(self, trajectory_str):
        """
        Description:
            A Bézier curve is a parametric curve used in computer graphics and related fields. 
            The function shows several types of smoothing trajectories using Bézier curves (Quadratic, Cubic).

        Args:
            (1) trajectory_str [Structure Type Array]: Structure of the trajectory points.

        Return:
            (1 - 2) parameter{1}, parameter{2} [Float Array]: Results of trajectory values (x, y).
            (3) parameter{3} [INT Array]: Results of trajectory values (Inverse Kinematics config).
        """

        try:
            assert len(trajectory_str) == 2 or len(trajectory_str) == 3

            if len(trajectory_str) == 2:
                cfg  = []
                time = np.linspace(0.0, 1.0, trajectory_str[0]['step'] + trajectory_str[1]['step'])

                # Quadratic Bezier Curve
                # p(t) = ((1 - t)^2)*p_{0} + 2*t*(1 - t)*p_{1} + (t^2)*p_{2}, t ∈ [0, 1]
                x = ((1 - time)**2) * trajectory_str[0]['start_p'][0] + 2 * (1 - time) * time * trajectory_str[0]['target_p'][0] + (time**2) * trajectory_str[1]['target_p'][0]
                y = ((1 - time)**2) * trajectory_str[0]['start_p'][1] + 2 * (1 - time) * time * trajectory_str[0]['target_p'][1] + (time**2) * trajectory_str[1]['target_p'][1]
                
                cfg.append([trajectory_str[0]['cfg']] * trajectory_str[0]['step'])
                cfg.append([trajectory_str[1]['cfg']] * trajectory_str[1]['step'])
            else:
                cfg  = []
                time = np.linspace(0.0, 1.0, trajectory_str[0]['step'] + trajectory_str[1]['step'] + trajectory_str[2]['step'])

                # Cubic Bezier Curve
                # p(t) = ((1 - t)^3)*p_{0} + 3*t*((1 - t)^2)*p_{1} + (3*t^2)*(1 - t)*p_{2} + (t^3) * p_{3}, t ∈ [0, 1]
                x = ((1 - time)**3) * (trajectory_str[0]['start_p'][0]) + (3 * time * (1 - time)**2) * (trajectory_str[0]['target_p'][0]) + 3 * (time**2) * (1 - time) * trajectory_str[1]['target_p'][0] + (time**3) * trajectory_str[2]['target_p'][0]
                y = ((1 - time)**3) * (trajectory_str[0]['start_p'][1]) + (3 * time * (1 - time)**2) * (trajectory_str[0]['target_p'][1]) + 3 * (time**2) * (1 - time) * trajectory_str[1]['target_p'][1] + (time**3) * trajectory_str[2]['target_p'][1]
                
                cfg.append([trajectory_str[0]['cfg']] * trajectory_str[0]['step'])
                cfg.append([trajectory_str[1]['cfg']] * trajectory_str[1]['step'])
                cfg.append([trajectory_str[2]['cfg']] * trajectory_str[2]['step'])

            self.plt.plot(x, y, '--', c=[0.1, 0.0, 0.7, 1.0], linewidth=3.0)

            return [x, y, np.concatenate(cfg)]

        except AssertionError:
            print('[INFO] Insufficient number of entry points.')
            print('[INFO] The number of entry points must be 3 (Quadratic Curve) or 4 (Cubic Curve).')

            return False

    def check_target(self, kinematics_str):
        """
        Description:
            Function to check whether the point is reachable or not. The function allows to check the input structure for Cartesian and Joint parameters.

        Args:
            (1) kinematics_str [Structure Type]: Structure of the trajectory point.

        Return:
            (1) param_1 [Bool]: The point is reachable or not.
        """

        if kinematics_str['calc_type'] == 'IK': 
            self.inverse_kinematics([kinematics_str['p'][0], kinematics_str['p'][1]], kinematics_str['cfg'])
        
            if (self.theta[0] < self.ax_wr[0] or self.theta[0] > self.ax_wr[1]) or (self.theta[1] < self.ax_wr[2] or self.theta[1] > self.ax_wr[3]):
                print('[INFO] Calculation Error [The target position (Joint representation) is outside of the working range].')
                print('[INFO] Calculation Error [Working range: Theta_1 (%f, %f), Theta_2(%f, %f)]:' % (self.ax_wr[0] * (180/np.pi), self.ax_wr[1]*(180/np.pi), self.ax_wr[2]*(180/np.pi), self.ax_wr[3]*(180/np.pi)))
                print('[INFO] Calculation Error [Target Position: Joint]:', self.theta[0]*(180/np.pi), self.theta[1]*(180/np.pi))

                # Mark error points with -> [*].
                self.plt.plot(kinematics_str['p'][0], kinematics_str['p'][1], marker = '*', ms = 10, mfc = [1,0,0], markeredgecolor = [0,0,0], mew = 2.5)

                return True
            else:
                if (round(self.__p_target[0], 3) != round(self.p[0], 3)) and (round(self.__p_target[1], 3) != round(self.p[1], 3)):
                    print('[INFO] Calculation Error [The target position (Cartesian representation) is outside of the workspace].')
                    print('[INFO] Calculation Error [Actual Position: End-Effector]:', round(self.p[0], 3), round(self.p[1], 3))
                    print('[INFO] Calculation Error [Target Position: End-Effector]:', round(self.__p_target[0], 3), round(self.__p_target[1], 3))

                    # Mark error points with -> [*].
                    self.plt.plot(kinematics_str['p'][0], kinematics_str['p'][1], marker = '*', ms = 10, mfc = [1,0,0], markeredgecolor = [0,0,0], mew = 2.5)

                    return True
                else:
                    return False

            self.__p_target = [kinematics_str['p'][0], kinematics_str['p'][1]]

        elif kinematics_str['calc_type'] == 'FK':
            theta_aux = np.zeros(2)

            if kinematics_str['degree_repr'] == True:
                theta_aux = [x * (np.pi/180) for x in kinematics_str['theta']]
            else:
                theta_aux = kinematics_str['theta']

            if (theta_aux[0] < self.ax_wr[0] or theta_aux[0] > self.ax_wr[1]) or (theta_aux[1] < self.ax_wr[2] or theta_aux[1] > self.ax_wr[3]):
                print('[INFO] Calculation Error [The target position (Joint representation) is outside of the working range].')
                print('[INFO] Calculation Error [Working range: Theta_1 (%f, %f), Theta_2(%f, %f)]:' % (self.ax_wr[0] * (180/np.pi), self.ax_wr[1]*(180/np.pi), self.ax_wr[2]*(180/np.pi), self.ax_wr[3]*(180/np.pi)))
                print('[INFO] Calculation Error [Target Position: Joint]:', theta_aux[0]*(180/np.pi), theta_aux[1]*(180/np.pi))

                return True
            else:
                return False

            self.__p_target = np.zeros(2)

        self.forward_kinematics(1, [0.0, 0.0], False)

    @staticmethod
    def __get_item_str(structure):
        """
        Description:
            The function of retrieving items from a structure.

        Args:
            (1) structure [Structure Type]: Different types of structures -> x = {'calc_type': 'IK', 'p': [0.20, 0.60], 'cfg': 0}

        Return:
            (1) param_1 [String Array]: Return items from the structure -> item = ['calc_type', 'p', 'cfg'] 

        Example:
           self.__get_item_str(x) 
        """

        item = []
        for x, i in structure.items():
            item.append(x)

        return item

    def check_trajectory(self, kinematics_str):
        """
        Description:
            Function for checking all reachable trajectory points.

        Args:
            (1) kinematics_str [Structure Type Array]: Structure of the trajectory points.
        
        Return:
            (1) param_{1,2} [Bool Array]: 1 - Reachable points error (True -> NOK, False -> OK), 2 - Index
        """

        err_p = [[], []]

        for i in range(len(kinematics_str)):
            aux_item = self.__get_item_str(kinematics_str[i])
            aux_kinematics_str = {aux_item[0]: kinematics_str[i][aux_item[0]], aux_item[1]: kinematics_str[i][aux_item[1]], aux_item[2]: kinematics_str[i][aux_item[2]]}

            err_p[0].append(self.check_target(aux_kinematics_str))
            err_p[1].append(i)

        return err_p

    def __generate_workspace_curve(self, limit, fk_param, increment):
        """
        Description:
           Simple function to generate a workspace curve.

        Args:
            (1) limit [Float Array]: Maximum and minimum limit of the curve.
            (2) fk_param [INT, Float Array]: Dynamic parameter for points from the FK calculation and the index of the current parameter.
            (3) increment [INT]: Number of increments and direction.

        Return:
            (1 - 2) parameter{1}, parameter{2} [Float Array]: Results of path values.
        """

        x = []
        y = []

        for i in range(int(limit[0] * (180/np.pi)), int(limit[1] * (180/np.pi)), increment):
            if fk_param[0] == 0:
                self.forward_kinematics(1, [fk_param[1], i * (np.pi/180)], False)
            elif fk_param[0] == 1:
                self.forward_kinematics(1, [i * (np.pi/180), fk_param[1]], False)

            x.append(self.p[0])
            y.append(self.p[1])

        return x, y

    def __display_workspace(self, display_type = 0):
        """
        Description:
            Display the work envelope (workspace) in the environment.

        Args:
            (1) display_type [INT]: Work envelope visualization options (0: Line, 1: Points).

        Examples:
            self._display_workspace(0)
        """

        # Generate linearly spaced vectors for the each of joints.
        theta_1 = np.linspace(self.ax_wr[0], self.ax_wr[1], 100)
        theta_2 = np.linspace(self.ax_wr[2], self.ax_wr[3], 100)

        # Return coordinate matrices from coordinate vectors.
        [theta_1_mg, theta_2_mg] = np.meshgrid(theta_1, theta_2)

        # Find the points x, y in the workspace using the equations FK.
        x_p = (self.rDH_param.a[0]*np.cos(theta_1_mg) + self.rDH_param.a[1]*np.cos(theta_1_mg + theta_2_mg))
        y_p = (self.rDH_param.a[0]*np.sin(theta_1_mg) + self.rDH_param.a[1]*np.sin(theta_1_mg + theta_2_mg))

        if display_type == 0:
            x_pN = []
            y_pN = []

            # Inner Circle -> Part 1
            x, y = self.__generate_workspace_curve([self.ax_wr[1], self.ax_wr[0]], [1, self.ax_wr[2]], -1)
            x_pN.append(x)
            y_pN.append(y)
            self.plt.plot(x_pN[0], y_pN[0], '-', c=[0,1,0,0.5], linewidth=5)
            # Inner Circle -> Part 2
            x, y = self.__generate_workspace_curve([self.ax_wr[1], self.ax_wr[0]], [1, self.ax_wr[3]], -1)
            x_pN.append(x)
            y_pN.append(y)
            self.plt.plot(x_pN[1], y_pN[1], '-', c=[0,1,0,0.5], linewidth=5)
            # Outer Curve -> Part 1
            x, y = self.__generate_workspace_curve([self.ax_wr[3], 0.0], [0, self.ax_wr[1]], -1)
            x_pN.append(x)
            y_pN.append(y)
            self.plt.plot(x_pN[2], y_pN[2], '-', c=[0,1,0,0.5], linewidth=5)
            # Outer Circle
            x, y = self.__generate_workspace_curve([self.ax_wr[1], self.ax_wr[0]], [1, 0.0], -1)
            x_pN.append(x)
            y_pN.append(y)
            self.plt.plot(x_pN[3], y_pN[3], '-', c=[0,1,0,0.5], linewidth=5)
            # Outer Curve -> Part 2
            x, y = self.__generate_workspace_curve([0.0, self.ax_wr[2]], [0, self.ax_wr[0]], -1)
            x_pN.append(x)
            y_pN.append(y)
            self.plt.plot(x_pN[4], y_pN[4], '-', c=[0,1,0,0.5], linewidth=5)
            
            self.plt.plot(x_p[0][0], y_p[0][0],'.', label=u"Work Envelop", c=[0,1,0,0.5])

        elif display_type == 1:
            self.plt.plot(x_p, y_p,'o', c=[0,1,0,0.1])
            self.plt.plot(x_p[0][0],y_p[0][0], '.', label=u"Work Envelop", c=[0,1,0,0.5])

    def __animation_data_generation(self):
        """
        Description:
            Generation data for animation. The resulting matrix {self.__animation_dMat} will contain each of the positions for the animation objects.
        """

        self.__animation_dMat = np.zeros((len(self.trajectory[0]), 4), dtype=np.float) 
        
        for i in range(len(self.trajectory[0])):
            self.inverse_kinematics([self.trajectory[0][i], self.trajectory[1][i]], self.trajectory[2][i])
            self.__animation_dMat[i][0] = self.rDH_param.a[0]*np.cos(self.rDH_param.theta[0])
            self.__animation_dMat[i][1] = self.rDH_param.a[0]*np.sin(self.rDH_param.theta[0])
            self.__animation_dMat[i][2] = self.p[0]
            self.__animation_dMat[i][3] = self.p[1]

    def init_animation(self):
        """
        Description: 
            Initialize each of the animated objects that will move in the animation.

        Return:
            (1) param_1 [Float Array]: Arrays of individual objects 
        """

        # Generation data for animation
        self.__animation_data_generation()

        # Initialization of robot objects
        self.__line[0].set_data([0.0, self.__animation_dMat[0][0]], [0.0, self.__animation_dMat[0][1]])
        self.__line[1].set_data([self.__animation_dMat[0][0], self.__animation_dMat[0][2]], [self.__animation_dMat[0][1], self.__animation_dMat[0][3]])
        self.__line[2].set_data(self.__animation_dMat[0][0], self.__animation_dMat[0][1])
        self.__line[3].set_data(self.__animation_dMat[0][2], self.__animation_dMat[0][3])
        
        return self.__line

    def start_animation(self, i):
        """
        Description: 
            Animation of the movement of a robot that runs until the end of the trajectory position.

        Args:
            (1) i [INT]: Iteration of the trajecotry.

        Return:
            (1) param_1 [Float Array]: Arrays of individual objects 
        """

        self.__line[0].set_data([0.0, self.__animation_dMat[i][0]], [0.0, self.__animation_dMat[i][1]])
        self.__line[1].set_data([self.__animation_dMat[i][0], self.__animation_dMat[i][2]], [self.__animation_dMat[i][1], self.__animation_dMat[i][3]])
        self.__line[2].set_data(self.__animation_dMat[i][0], self.__animation_dMat[i][1])
        self.__line[3].set_data(self.__animation_dMat[i][2], self.__animation_dMat[i][3])

        return self.__line

    def display_environment(self, work_envelope = [False, 0]):
        """
        Description:
            Display the entire environment with the robot and other functions (such as a work envelope).

        Args:
            (1) work_envelope [Array (BOOL, INT)]: Work Envelop options (Visibility, Type of visualization (0: Line, 1: Points)).

        Examples:
            self.display_environment([True, 0])
        """

        # Condition for visible work envelop
        if work_envelope[0] == True:
            self.__display_workspace(work_envelope[1])

        if len(self.trajectory[0]) > 0:
            self.plt.plot(self.trajectory[0][0], self.trajectory[1][0], label=r'Trajectory Points: $p_{(x, y)}$', marker = 'o', ms = 15, mfc = [1,1,0], markeredgecolor = [0,0,0], mew = 5)
            self.plt.plot(self.trajectory[0][0], self.trajectory[1][0], label=r'Initial Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [0,0.5,1], markeredgecolor = [0,0,0], mew = 5)
            self.plt.plot(self.trajectory[0][len(self.trajectory[0]) - 1], self.trajectory[1][len(self.trajectory[1]) - 1], label=r'Target Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [0,1,0], markeredgecolor = [0,0,0], mew = 5)

            self.p = [self.trajectory[0][len(self.trajectory[0]) - 1], self.trajectory[1][len(self.trajectory[1]) - 1]]
            self.inverse_kinematics(self.p, self.trajectory[2][len(self.trajectory[0]) - 1])
       
        # Display FK/IK calculation results (depens on type of options)
        self.__display_result()

        # Set minimum / maximum environment limits
        self.plt.axis([(-1)*(self.rDH_param.a[0] + self.rDH_param.a[1]) - 0.2, (1)*(self.rDH_param.a[0] + self.rDH_param.a[1]) + 0.2, (-1)*(self.rDH_param.a[0] + self.rDH_param.a[1]) - 0.2, (1)*(self.rDH_param.a[0] + self.rDH_param.a[1]) + 0.2])

        # Set additional parameters for successful display of the robot environment
        self.plt.grid()
        self.plt.xlabel('x position [m]', fontsize = 20, fontweight ='normal')
        self.plt.ylabel('y position [m]', fontsize = 20, fontweight ='normal')
        self.plt.title(self.robot_name, fontsize = 50, fontweight ='normal')
        self.plt.legend(loc=0,fontsize=20)

    def __display_rDHp(self):
        """
        Description: 
            Display the DH robot parameters.
        """

        print('[INFO] The Denavit-Hartenberg modified parameters of robot %s:' % (self.robot_name))
        print('[INFO] theta = [%f, %f]' % (self.rDH_param.theta[0], self.rDH_param.theta[1]))
        print('[INFO] a     = [%f, %f]' % (self.rDH_param.a[0], self.rDH_param.a[1]))
        print('[INFO] d     = [%f, %f]' % (self.rDH_param.d[0], self.rDH_param.d[1]))
        print('[INFO] alpha = [%f, %f]' % (self.rDH_param.alpha[0], self.rDH_param.alpha[1]))

    def __display_result(self):
        """
        Description: 
            Display of the result of the robot kinematics (forward/inverse) and other parameters.
        """

        print('[INFO] Result of Kinematics calculation:')
        print('[INFO] Robot: %s' % (self.robot_name))

        if not (self.__p_target is None):
            print('[INFO] Target Position (End-Effector):')
            print('[INFO] p_t  = [x: %f, y: %f]' % (self.__p_target[0], self.__p_target[1]))
            print('[INFO] Target Position (Joint):')
            print('[INFO] Theta  = [Theta_1: %f, Theta_1: %f]' % (self.__theta_target[0], self.__theta_target[1]))
            print('[INFO] Actual Position (End-Effector):')
            print('[INFO] p_ee = [x: %f, y: %f]' % (self.p[0], self.p[1]))
            print('[INFO] Actual Position (Joint):')
            print('[INFO] Theta  = [Theta_1: %f, Theta_1: %f]' % (self.theta[0], self.theta[1]))
        else:
            print('[INFO] Target Position (Joint):')
            print('[INFO] Theta  = [Theta_1: %f, Theta_1: %f]' % (self.__theta_target[0], self.__theta_target[1]))
            print('[INFO] Actual Position (End-Effector):')
            print('[INFO] p_ee = [x: %f, y: %f]' % (self.p[0], self.p[1]))