import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/scu2u/senior_design/senior_design/ros2_ws/install/python_package'
