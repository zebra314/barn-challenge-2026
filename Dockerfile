FROM ros:melodic

# Set environment variables
ENV TERM=xterm-256color
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=melodic

# Install system dependencies
RUN apt-get update
RUN apt-get install -y python-pip
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-dev
RUN apt-get install -y git
RUN apt-get install -y curl
RUN apt-get install -y zsh
RUN chsh -s $(which zsh)

# Install GUI dependencies
RUN apt-get install -y libpci-dev
RUN apt-get install -y x11-apps
RUN apt-get install -y qtwayland5
RUN apt-get install -yqq xserver-xorg
RUN apt-get install -y xwayland

# Install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install p10k
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="powerlevel10k\/powerlevel10k"/g' ~/.zshrc
COPY dotfiles/.p10k.zsh /root/.p10k.zsh
COPY dotfiles/.zshrc /root/.zshrc

# Install multiple zsh plugins
# 1. zsh-autosuggestions
# 2. zsh-syntax-highlighting
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
RUN sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/g' ~/.zshrc

# Create ROS workspace
WORKDIR /jackal_ws/src

# Clone required ros packages
RUN git clone https://github.com/jackal/jackal.git --branch melodic-devel
RUN git clone https://github.com/jackal/jackal_simulator.git --branch melodic-devel
RUN git clone https://github.com/jackal/jackal_desktop.git --branch melodic-devel
RUN git clone https://github.com/utexas-bwi/eband_local_planner.git

# Install Python dependencies
RUN pip install "numpy<1.17"  pathlib "casadi==3.5.5"
RUN pip3 install numpy rospkg

# Install specific ROS packages
RUN apt-get install -y ros-melodic-rviz
RUN apt-get install -y ros-melodic-tf2-ros
RUN apt-get install -y ros-melodic-xacro
RUN apt-get install -y ros-melodic-twist-mux
RUN apt-get install -y ros-melodic-velodyne-description
RUN apt-get install -y ros-melodic-hector-gazebo-plugins
RUN apt-get install -y ros-melodic-rosdoc-lite
RUN apt-get install -y ros-melodic-base-local-planner
RUN apt-get install -y ros-melodic-navigation
RUN apt-get install -y ros-melodic-control-toolbox
RUN apt-get install -y ros-melodic-tf2-eigen
RUN apt-get install -y ros-melodic-lms1xx
RUN apt-get install -y ros-melodic-pointgrey-camera-description
RUN apt-get install -y ros-melodic-interactive-marker-twist-server
RUN apt-get install -y ros-melodic-robot-localization
RUN apt-get install -y ros-melodic-controller-manager
RUN apt-get install -y ros-melodic-sick-tim
RUN apt-get install -y ros-melodic-jackal-gazebo

# Install ROS package dependencies
WORKDIR /jackal_ws
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && \
    (rosdep init || echo 'rosdep already initialized') && \
    rosdep update && \
    rosdep install -y --from-paths src --ignore-src --rosdistro=melodic \
    --skip-keys='rviz tf2_ros xacro twist_mux velodyne_description hector_gazebo_plugins rosdoc_lite' || true"

# Clean
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["zsh"]
