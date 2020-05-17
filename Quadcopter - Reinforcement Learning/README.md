# Quadcopter - Reinforcement Learning
This project applies Reinforcement Learning algorithm to design an agent to fly a quadcopter.  The implementation is inspired by the methodology in the original [Deep Deterministic Policy Gradient (DDPG) paper](https://arxiv.org/abs/1509.02971). This is based on the actor critic model where both are implemented using deep neural network. The task is to train the quadcopter to fly from an initial location to a target location. 

### Project Files

The project uses these files and folders:

- [Quadcopter_Project.ipynb](https://github.com/udaygoel/Deep-Learning-Udacity/blob/master/Quadcopter%20-%20Reinforcement%20Learning/Quadcopter_Project.ipynb): This Jupyter Notebook covers the project.  

- physics_sim.py: This file contains the simulator for the quadcopter. This is provided by Udacity and should not be modified.

- task.py: File to define the task (environment) for the implementation.

- agents/agent.py: The reinforcement learning agent for this project.

- agents/policy_search.py: A sample reinforcement learning agent.

- data.txt: The output from a basic agent to understand how the quadcopter is working

  

### Contents

There are 5 main sections of the Project.

1. Controlling the Quadcopter 

   This is an illustration to see how the quadcopter works and the states evolve with each step, by creating a basic agent that selects a random action for each of the four rotors.

2. Define the Task and the Agent

   These are defined in their respective .py files. The implementation is based on the [Deep Deterministic Policy Gradient (DDPG) paper](https://arxiv.org/abs/1509.02971). The reward function is declared in the task.py file.

3. Training the Agent

   The agent is trained for 2000 episodes. The episodes can end early if the agent completes the task of reaching the target destination or it crashes. The agent receives an award for each step.

5. Display Results

   The path of the quadcopter across x and y axis is plotted for a subset of episodes.
   
5. Reflections

   This section explains the steps taken to build and train the network. Though, the network isn't completely trained, this section highlights the challenges faced and provides ideas for further improvement.
