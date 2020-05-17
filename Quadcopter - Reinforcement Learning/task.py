import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Testing inputs for correctness
        self.input_check(init_pose)
        self.input_check(target_pos)
        
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        self.last_sim_pos = None
        self.cosine_vectors_angle = None
        self.done_override = None
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        dir_target_pos = self.target_pos - init_pose[:3]
        self.init_radial_distance = np.sqrt(dir_target_pos.dot(dir_target_pos))

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        self.done_override = None
        reward = 0
        
        dir_target_pos = self.target_pos - self.sim.pose[:3]
        radial_distance = np.sqrt(dir_target_pos.dot(dir_target_pos))
        
        # reward for distance from target
        #radial_distance = np.sqrt(np.sum(np.square(self.sim.pose[:3] - self.target_pos)))
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = - 0.02* radial_distance
        #reward += np.exp(-1.2*radial_distance)
        
        # reward for reaching close to target
        if radial_distance < 0.5:
            self.done_override = True
            reward += 20
        
        # penalise for crashing to ground
        if self.last_sim_pos is None:
            self.last_sim_pos = self.sim.pose
        elif self.last_sim_pos[2] >= 0.0 and self.sim.pose[2] < 0.0:
            reward += -1.
            self.done_override = True
        """    
        if self.sim.pose[2] < 0.0:
            reward += -50.
        """
                
        # reward for direction of travel
        # finding the cosine of the angle between the velocity vector and the vector representing the shortest direction to the target
        # if the two vectors are in same direction, then cosine is 1. If they are orthogonal, cosine is 0 and if they are in 
        # opposite direction, then cosine is -1. Penalizing the agent if the direction of the two vectors is not same. Due to the 
        # nature of cosine, this is a non-linear function and gives less penalty even if the angle between the vectors is 15 degrees.
        # Cosine of 15 degrees ~ 0.965. The penalty increases non-linearly as the angle increases
        
        velocity = self.sim.v
        magnitude_velocity = np.sqrt(velocity.dot(velocity))
        self.cosine_vectors_angle = (dir_target_pos.dot(velocity)) / (radial_distance * magnitude_velocity)
        #reward += 0.15 *(self.cosine_vectors_angle-0.5)*(1+np.exp(-1.2*radial_distance)) + \
        #          0.2*max(0.0, self.cosine_vectors_angle)*(1-np.exp(-0.5*radial_distance))
        
        #reward = np.exp(-1.2*radial_distance) + max(0.0, self.cosine_vectors_angle - 0.5)*(1-np.exp(-1.2*radial_distance))
        #reward = np.exp(-radial_distance) *max(0.0,self.cosine_vectors_angle - 0.25)
        reward = max(0.0,1. - np.sqrt(radial_distance / (self.init_radial_distance*1.10)))
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        self.last_sim_pos = None
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            #pose_all.append(self.sim.pose)
            pose_all.append(np.concatenate((self.sim.pose[:3], self.sim.v)))
        next_state = np.concatenate(pose_all)
        if self.done_override is not None:
            done = self.done_override
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat) 
        state = np.concatenate([np.concatenate((self.sim.pose[:3], self.sim.v))] * self.action_repeat) 
        return state
    
    def input_check(self, pos):
        """Testing the position state for negative values of z. Z needs to be non-negative"""
        if pos is not None:
            if pos[2] < 0.0:
                print('Error: Negative z value for the initial or target state')
                exit()
        return