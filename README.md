# Two-Link Manipulator: 2DOF ABB IRB 910SC (SCARA)

## Requirements:

**Programming Language:**

```bash
Python
```

**Import Libraries:**
```bash
Manipulator Control: Matplotlib, NumPy
Dynamics: Matplotlib, NumPy, SciPy
```

## Project Description:

The project is focused on the Demonstration of simple control of a two-link robotic arm (2 degrees of freedom) implemented in Python using animation. The main parameters of the robot (length of arms, working range, etc.) are from the ABB IRB 910SC robot (SCARA), but only for 2-DOF control.

The main idea of the project is adaptability to future automation technologies in various fields and improving literacy (understanding kinematics, dynamics, motion planning, etc., in the field of robotics). The project was created to improve the [VRM (Programming for Robots and Manipulators)](https://github.com/rparak/Programming-for-robots-and-manipulators-VRM) university course.

Main functions of the robotic arm (2-DOF):
- Inverse / Forward kinematics, Differential Kinematics (Jacobian)
- Creation of a working envelope (Points, Line)
- Motion planning using Joint / Cartesian interpolation
- Trajectory smoothing using Bézier curves
- Animation of the resulting trajectory
- Check of reachable points, etc.

The project was realized at the Institute of Automation and Computer Science, Brno University of Technology, Faculty of Mechanical Engineering (NETME Centre - Cybernetics and Robotics Division).

<p align="center">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/2.png width="700" height="500">
</p>

## Project Hierarchy:

**Repositary [/2-Link_Manipulator/]:**
```bash
[ A simple example of implementing euler-lagrange dynamics        ] /Script/Dynamics/
[ Main Script (Test) + Manipulator Control Class                  ] /Script/Manipulator/
[ IRB 910SC Product Specification                                 ] /Product_specification/
```

## Application:

**Work Envelop:**

<p align="center">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/8.png width="400" height="325">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/9.png width="400" height="325">
</p>

**Linear and Joint Interpolation:**

<p align="center">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/10.png width="400" height="325">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/11.png width="400" height="325">
</p>

**Trajectory problem detection (some points not reachable):**

<p align="center">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/6.png width="400" height="325">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/7.png width="400" height="325">
</p>

**Trajectory Generation (Circle, Rectangle):**

<p align="center">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/4.png width="400" height="325">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/5.png width="400" height="325">
</p>

**Trajectory smoothing using Bézier curves:**

<p align="center">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/1.png width="400" height="325">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/images/3.png width="400" height="325">
</p>

## Result:

<p align="center">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/GIF/Default_3.gif width="800" height="500">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/GIF/Default_3_Smooth.gif width="800" height="500">
</p>

<p align="center">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/GIF/Default_4.gif width="800" height="500">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/GIF/Default_4_Smooth.gif width="800" height="500">
</p>

<p align="center">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/GIF/Circle.gif width="800" height="500">
 <img src=https://github.com/rparak/2-Link_Manipulator/blob/main/GIF/Rectangle.gif width="800" height="500">
</p>

## Contact Info:
Roman.Parak@outlook.com

## Citation (BibTex)

```bash
@misc{RomanParak_2LArm,
  author = {Roman Parak},
  title = {Simple control of a two-link robotic arm implemented in Python},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rparak/Bioloid-Dynamixel-AX12A}}
}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
