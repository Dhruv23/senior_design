from setuptools import find_packages, setup

package_name = 'python_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='scu2u',
    maintainer_email='scu2u@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'log_imu_data_wifi = python_package.log_imu_data_wifi:main',
            'test_pub = python_package.testing.test_pub:main',
            'test_sub = python_package.testing.test_sub:main',
            'send_commands = python_package.send_commands:main',
        ],
    },
)
