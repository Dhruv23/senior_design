from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("scu2u_description")

    world_path = PathJoinSubstitution([pkg, "worlds", "empty.sdf"])
    urdf_path  = PathJoinSubstitution([pkg, "urdf", "rover.urdf"])

    use_sim_time = LaunchConfiguration("use_sim_time")

    # Read URDF file contents into robot_description
    robot_description = Command(["cat ", urdf_path])

    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", world_path],
        output="screen",
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": use_sim_time
        }],
        output="screen",
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "rover",
            "-string", robot_description,
            "-x", "0.0", "-y", "0.0", "-z", "0.2",
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        gazebo,
        rsp,
        spawn,
    ])
