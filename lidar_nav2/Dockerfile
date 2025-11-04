# Base: ROS 2 Humble on Ubuntu 22.04
FROM ros:humble

# Install Python + plotting deps
RUN apt-get update && apt-get install -y \
    python3-pip python3-numpy python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*

# Copy your node and dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rplidar_nav2_node.py .

# Source ROS and run node
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && python3 /app/rplidar_nav2_node.py"]
