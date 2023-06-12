# Use the official Miniconda3 image as the base image
FROM continuumio/miniconda3

# Copy the current directory contents into the container at /app

# # Create a new Conda environment with Python 3
RUN conda create -n torch python=3.10

# # Activate the Conda environment
SHELL ["conda", "run", "-n", "torch", "/bin/bash", "-c"]

# # Install PyTorch and torchvision
RUN conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia -y
RUN conda install numpy matplotlib dill -y

RUN git clone https://github.com/KaiMJ/neural_climate_processes.git

# Set the working directory to /app
WORKDIR /neural_climate_processes

# # Set the default command to run when the container starts
# COPY . .
ENTRYPOINT [ "conda", "run", "-n", "torch"]
WORKDIR /neural_climate_processes/src
CMD ["python", "run.py", "test"]
# CMD ["conda", "run", "-n", "torch", "python", "main.py"]