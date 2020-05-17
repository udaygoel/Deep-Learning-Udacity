# TODO: your agent here!

from keras import layers, models, optimizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Lambda
import random
from collections import namedtuple, deque
import numpy as np
import copy

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, actor_learning_rate = 0.01,
                 actor_num_hidden_units = None, actor_dropout_rate = 0.2, states_scaler = None, actions_scaler = None):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.learning_rate = actor_learning_rate
        self.dropout_rate = actor_dropout_rate
        self.num_hidden_units = actor_num_hidden_units
        
        # Setting up scalers
        self.states_scaler = states_scaler
        self.actions_scaler = actions_scaler
        
        self.build_model()
        
        return

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        for ii in range(len(self.num_hidden_units)):
            
            if ii == 0:
                input_tf = states
                #net = layers.BatchNormalization()(input_tf)
            else:
                input_tf = net
            
            net = layers.Dense(units = self.num_hidden_units[ii], activation = None)(input_tf)
            net = layers.BatchNormalization()(net)
            net = layers.LeakyReLU(alpha = 0.1)(net)
            net = layers.Dropout(self.dropout_rate)(net)
             
        #net = layers.Dense(units=32, activation='relu')(states)
        #net = layers.Dense(units=64, activation='relu')(net)
        #net = layers.Dense(units=32, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        #model.add(Dense(units=self.action_size, activation='sigmoid', name='raw_actions'))
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        # 0.95 scaling added to avoid the min and max values and add learning
        #model.add(Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions'))
        #actions = layers.Lambda(lambda x: self.actions_scaler.inverse_transform(x),
        #    name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=raw_actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * raw_actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr = self.learning_rate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
        
        print(self.model.summary())
        return
    

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, critic_learning_rate = 0.01, critic_num_hidden_units = None, critic_dropout_rate = 0.2,
                states_scaler = None, actions_scaler = None):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.learning_rate = critic_learning_rate
        self.dropout_rate = critic_dropout_rate
        self.num_hidden_units = critic_num_hidden_units
        
        # Setting up scalers
        self.states_scaler = states_scaler
        self.actions_scaler = actions_scaler
        
        self.build_model()
        
        return
    
    def build_nn(self, input_val):
        for ii in range(len(self.num_hidden_units)):
            if ii == 0:
                input_tf = input_val
                #net = layers.BatchNormalization()(input_tf)
            else:
                input_tf = net
                
            net = layers.Dense(units = self.num_hidden_units[ii], activation = None)(input_tf)
            net = layers.BatchNormalization()(net)
            net = layers.Activation('relu')(net)
            net = layers.Dropout(self.dropout_rate)(net)
        return net
         

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = self.build_nn(states)
               
        #net_states = layers.Dense(units=32, activation='relu')(states)
        #net_states = layers.Dense(units=64, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = self.build_nn(actions)
       
        #net_actions = layers.Dense(units=32, activation='relu')(actions)
        #net_actions = layers.Dense(units=64, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        net = layers.Dense(units = 64, activation = 'relu')(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr = self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
        
        print(self.model.summary())
        
        return
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        return

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        return

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        return

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        return

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, gamma = 0.99, tau = 0.01, buffer_size= 100000, batch_size = 64, noise_mu = 0.0,
                 noise_theta = 0.15, noise_sigma = 0.2, actor_learning_rate = 0.01, 
                 actor_num_hidden_units = None, actor_dropout_rate = 0.2,
                 critic_learning_rate = 0.01, critic_num_hidden_units = None, critic_dropout_rate = 0.2, 
                states_scaler = None, actions_scaler = None):
        
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        print ('\nActor Local Model')
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, 
                                actor_learning_rate, actor_num_hidden_units, actor_dropout_rate, states_scaler, actions_scaler)
        print ('\nActor Target Model')
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                 actor_learning_rate, actor_num_hidden_units, actor_dropout_rate, states_scaler, actions_scaler)

        # Critic (Value) Model
        print ('\nCritic Local Model')
        self.critic_local = Critic(self.state_size, self.action_size, critic_learning_rate,
                                   critic_num_hidden_units, critic_dropout_rate, states_scaler, actions_scaler)
        print ('\nCritic Target Model')
        self.critic_target = Critic(self.state_size, self.action_size, critic_learning_rate,
                                   critic_num_hidden_units, critic_dropout_rate, states_scaler, actions_scaler)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_loss = None

        # Noise process
        self.exploration_mu = noise_mu
        self.exploration_theta = noise_theta
        self.exploration_sigma = noise_sigma
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        self.noise_sample = None

        # Replay memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters
        
        # Setting up scalers
        self.states_scaler = states_scaler
        self.actions_scaler = actions_scaler
        return

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        state = self.states_scaler.transform([state])[0]
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        # Scale action and next_state using scalers
        action = self.actions_scaler.transform([action])[0]
        next_state = self.states_scaler.transform([next_state])[0]
        
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state
        return 

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        action = self.actions_scaler.inverse_transform([action])[0]
        self.noise_sample = self.noise.sample()
        action_floored = np.maximum((action + self.noise_sample), 
                                    np.ones(self.action_size)*(self.action_low + 0.01))
        action_bound = np.minimum(action_floored, np.ones(self.action_size)*self.action_high)
        return list(action_bound)  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_loss = self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   
        return

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        return