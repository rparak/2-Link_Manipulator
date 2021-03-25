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
File Name: main.py
## =========================================================================== ## 
"""

# System (Default Lib.)
import sys
# Own library for robot control (kinematics), visualization, etc. (See manipulator.py)
import manipulator
# Matplotlib (Visualization Lib. -> Animation) [pip3 install matplotlib]
from matplotlib import animation
# Numpy (Array computing Lib.) [pip3 install numpy]
import numpy as np

def generate_rectangle(centroid, dimension, angle):
    """
    Description:
        A simple function to generate a path for a rectangle.

    Args:
        (1) centroid [Float Array]: Centroid of the Rectangle (x, y).
        (2) dimension [Float Array]: Dimensions (width, height).
        (3) angle [Float]: Angle (Degree) of the Rectangle.
        
    Returns:
        (1 - 2) parameter{1}, parameter{2} [Float Array]: Results of path values.

    Examples:
        generate_rectangle([1.0, 1.0], [1.0, 1.0], 0.0)
    """

    p = [[(-1)*dimension[0]/2, (-1)*dimension[0]/2, (+1)*dimension[0]/2, (+1)*dimension[0]/2, (-1)*dimension[0]/2],
         [(-1)*dimension[1]/2, (+1)*dimension[1]/2, (+1)*dimension[1]/2, (-1)*dimension[1]/2, (-1)*dimension[1]/2]]

    x = []
    y = []

    for i in range(len(p[0])):
        # Calculation position of the Rectangle
        x.append((p[0][i]*np.cos(angle * (np.pi/180)) - p[1][i]*np.sin(angle * (np.pi/180))) + centroid[0])
        y.append((p[0][i]*np.sin(angle * (np.pi/180)) + p[1][i]*np.cos(angle * (np.pi/180))) + centroid[1])

    return [x, y]

def generate_circle(centroid, radius):
    """
    Description:
        A simple function to generate a path for a circle.

    Args:
        (1) centroid [Float Array]: Centroid of the Circle (x, y).
        (1) radius [Float]: Radius of the Circle (r).
    Returns:
        (1 - 2) parameter{1}, parameter{2} [Float Array]: Results of path values.

    Examples:
        generate_circle([1.0, 1.0], 0.1)
    """

    # Circle ->  0 to 2*pi
    theta = np.linspace(0, 2*np.pi, 25)

    # Calculation position of the Circle
    x = radius * np.cos(theta) + centroid[0]
    y = radius * np.sin(theta) + centroid[1]

    return [x, y]

def main():
    # Initial Parameters -> ABB IRB910SC 
    # Product Manual: https://search.abb.com/library/Download.aspx?DocumentID=3HAC056431-001&LanguageCode=en&DocumentPartId=&Action=Launch

    # Working range (Axis 1, Axis 2)
    axis_wr = [[-140.0, 140.0],[-150.0, 150.0]]
    # Length of Arms (Link 1, Link2)
    arm_length = [0.3, 0.25]

    # DH (Denavit-Hartenberg) parameters
    theta_0 = [0.0,0.0]
    a       = [arm_length[0], arm_length[1]]
    d       = [0.0, 0.0]
    alpha   = [0.0, 0.0]

    # Initialization of the Class (Control Manipulator)
    # Input:
    #   (1) Robot name         [String]
    #   (2) DH Parameters      [DH_parameters Structure]
    #   (3) Axis working range [Float Array]
    scara = manipulator.Control('ABB IRB 910SC (SCARA)', manipulator.DH_parameters(theta_0, a, d, alpha), axis_wr)

    """
    Example (1): 
        Description:
            Chack Target (Point) -> Check that the goal is reachable for the robot

        Cartesian Target:
            x = {'calc_type': 'IK', 'p': [0.20, 0.60], 'cfg': 0}
        Joint Target:
            x = {'calc_type': 'FK', 'theta': [0.0, 155.0], 'degree_repr': True}

        Call Function:
            res = scara.check_target(check_cartesianTarget)

    Example (2): 
        Description:
            Test Results of the kinematics.

        Forward Kinematics:
            x.forward_kinematics(1, [0.0, 0.0], True)
        Inverse Kinematics:
            (a) Default Calculation method
                x.inverse_kinematics([0.35, 0.15], 1)
            (b) Jacobian Calculation method
                x.inverse_kinematics_jacobian([0.35, 0.15], [0.0, 0.0], 0.0001, 10000)
        Both kinematics to each other:
            x.forward_kinematics(0, [0.0, 45.0], True)
            x.inverse_kinematics(x.p, 1)
    """

    # Test Trajectory (Select one of the options - Create a trajectory structure -> See below)
    test_trajectory = 'Default_3'

    # Structure -> Null
    trajectory_str = []

    if test_trajectory == 'Circle':
        # Generating a trajectory structure for a circle
        # Input:
        #   (1) Circle Centroid [Float Array]
        #   (2) Radius          [Float]
        x, y = generate_circle([0.25, -0.25], 0.1)

        # Initial (Start) Position
        trajectory_str.append({'interpolation': 'joint', 'start_p': [0.50, 0.0], 'target_p': [x[0], y[0]], 'step': 100, 'cfg': 1})

        for i in range(len(x) - 1):
            trajectory_str.append({'interpolation': 'linear', 'start_p': [x[i], y[i]], 'target_p': [x[i + 1], y[i + 1]], 'step': 10, 'cfg': 1})

    elif test_trajectory == 'Rectangle':
        # Generating a trajectory structure for a rectangle
        # Input:
        #   (1) Rectangle Centroid         [Float Array]
        #   (2) Dimensions (width, height) [Float Array]
        #   (3) Angle (Degree)             [Float]
        x, y = generate_rectangle([-0.25, 0.25], [0.15, 0.15], 0.0)

        # Initial (Start) Position
        trajectory_str.append({'interpolation': 'joint', 'start_p': [0.50, 0.0], 'target_p': [x[0], y[0]], 'step': 100, 'cfg': 0})

        for i in range(len(x) - 1):
            trajectory_str.append({'interpolation': 'linear', 'start_p': [x[i], y[i]], 'target_p': [x[i + 1], y[i + 1]], 'step': 50, 'cfg': 0})

    elif test_trajectory == 'Default_1':
        # Initial (Start) Position
        trajectory_str.append({'interpolation': 'linear', 'start_p': [0.50, 0.0], 'target_p': [0.5, 0.0], 'step': 100, 'cfg': 0})

    elif test_trajectory == 'Default_2':
        # Generating a trajectory structure between two points
        trajectory_str.append({'interpolation': 'joint', 'start_p': [0.30, 0.0], 'target_p': [0.0, 0.40], 'step': 50, 'cfg': 1})

    elif test_trajectory == 'Default_3':
        # Generating a trajectory structure between three points
        trajectory_str.append({'interpolation': 'linear', 'start_p': [0.30, 0.0], 'target_p': [0.40, 0.30], 'step': 50, 'cfg': 1})
        trajectory_str.append({'interpolation': 'linear', 'start_p': [0.40, 0.30], 'target_p': [0.20, 0.40], 'step': 50, 'cfg': 1})

    elif test_trajectory == 'Default_4':
        # Generating a trajectory structure between four points
        trajectory_str.append({'interpolation': 'linear', 'start_p': [0.30, 0.0], 'target_p': [0.40, 0.30], 'step': 25, 'cfg': 1})
        trajectory_str.append({'interpolation': 'linear', 'start_p': [0.40, 0.30], 'target_p': [0.20, 0.40], 'step': 25, 'cfg': 1})
        trajectory_str.append({'interpolation': 'linear', 'start_p': [0.20, 0.40], 'target_p': [0.0, 0.30], 'step': 25, 'cfg': 1})
 
    # Structure -> Null
    check_cartesianTrajectory_str = []

    for i in range(len(trajectory_str)):
        # Generating a trajectory from a structure
        x, y, cfg = scara.generate_trajectory(trajectory_str[i])

        for j in range(trajectory_str[i]['step']):
            # Create a Cartesian trajectory structure for each of the points and configurations
            check_cartesianTrajectory_str.append({'calc_type': 'IK', 'p': [x[j], y[j]], 'cfg': cfg[j]})
            # Adding points and configurations to the resulting trajectory
            scara.trajectory[0].append(x[j])
            scara.trajectory[1].append(y[j])
            scara.trajectory[2].append(cfg[j])
    
    # Check that the trajectory points for the robot are reachable
    tP_err = scara.check_trajectory(check_cartesianTrajectory_str)

    # Smoothing the trajecotory using Bézier Curve (3 points -> Quadratic, 4 -> Points Cubic)
    tP_smooth = True

    if tP_smooth == True:
        smooth_trajectory = [[], [], []]
        try :
            # Check that trajectory smoothing is possible and smooth the trajectory using an appropriate method
            [smooth_trajectory[0], smooth_trajectory[1], smooth_trajectory[2]] = scara.smooth_trajectory(trajectory_str)
            # Trajectory smoothing is successful
            scara.trajectory = smooth_trajectory
        except TypeError:
            print('[INFO] Trajectory smoothing is not possible (Insufficient or too many entry points).')

    # 1. Display the entire environment with the robot and other functions.
    # 2. Display the work envelope (workspace) in the environment (depends on input).
    # Input:
    #  (1) Work Envelop Parameters
    #       a) Visible                   [BOOL]
    #       b) Type (0: Line, 1: Points) [INT]
    scara.display_environment([True, 1])

    if True in tP_err[0]:
        # Trajectory Error (some points are not not reachable)
        scara.init_animation()
    else:
        # Call the animator for the SCARA Robotics Arm (if the results of the solution are error-free).
        animator = animation.FuncAnimation(scara.figure, scara.start_animation, init_func=scara.init_animation, frames=len(scara.trajectory[0]), interval=25, blit=True, repeat=False)

    scara.plt.show()

if __name__ == '__main__':
    sys.exit(main())
