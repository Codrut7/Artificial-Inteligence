import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda')

class DQLModel(nn.Module):
    def __init__(self, observation_space_size, action_space_size, hidden_space_size=64):
        super(DQLModel, self).__init__()

        # Define the hyperparameters of the NN
        self.input_size = observation_space_size
        self.output_size = action_space_size
        self.hidden_size = hidden_space_size

        # Define the linear layers of the NN
        self.linear_1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_4 = nn.Linear(self.hidden_size, self.output_size)

        # Define the activation function
        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.linear_1(x)
        x = self.activation(x)
        
        x = self.linear_2(x)
        x = self.activation(x)

        x = self.linear_3(x)
        x = self.activation(x)

        x = self.linear_4(x)

        return x

class DQLAgent():
    def __init__(self, observation_space_size, action_space_size, enviroment):
        
        # space constants
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        # Batch size
        self.batch_size = 64
        # Exploration settings
        self.epsilon = 1
        self.epsilon_decay = 0.99975
        self.min_epsilon = 0.001
        self.update_target = 5
        self.discount = 0.99
        # Model used to predict each move of the snake
        self.prediction_model = DQLModel(observation_space_size, action_space_size).to(device)
        #self.prediction_model.load_state_dict(torch.load(r"D:\Projects\Reinforcement Learning\LearningSnake\model\target_model.pth"))
        # Target model used for the label prediction
        self.target_model = DQLModel(observation_space_size, action_space_size).to(device)
        #self.target_model.load_state_dict(torch.load(r"D:\Projects\Reinforcement Learning\LearningSnake\model\target_model.pth"))
        # Model training optimizer / loss
        self.optimizer = torch.optim.Adam(self.prediction_model.parameters(), lr=1e-4)
        self.loss_function = nn.MSELoss()
        # Enviroment
        self.env = enviroment

    def train(self, update_counter):
        
        batch = self.env.get_batch()
        
        # Get a batch of moves from the replay memory
        # If there are not enough moves return and use epsilon for exploration
        if batch is None:
            return
        
        # Get the current states from batch, then predict the Q values for each action of the state
        current_states = torch.FloatTensor([idx[0] for idx in batch]).view(self.batch_size, -1).to(device)
        current_qs_list = self.prediction_model(current_states)

        # Get the future states from batch, predict Q value
        done_state_mask = np.zeros(21)
        future_states = torch.FloatTensor([idx[3] if idx[3] is not None else done_state_mask for idx in batch]).view(self.batch_size, -1).to(device)
        future_qs_list = self.target_model(future_states)

        inp = torch.zeros(self.batch_size, self.observation_space_size).type(torch.FloatTensor).to(device)
        target = torch.zeros(self.batch_size, self.action_space_size).type(torch.FloatTensor).to(device)

        for idx, (current_state, action, reward, new_current_state, done) in enumerate(batch):

            if done:
                target_q = reward
            else:
                max_future_q = torch.max(future_qs_list[idx])
                target_q = reward + self.discount * max_future_q.item()

            current_q = current_qs_list[idx]
            current_q[action] = target_q

            # Current state is the input of the model
            inp[idx] = torch.FloatTensor(current_state).to(device)
            # Correct q value for the best action is the output target
            target[idx] = current_q

        self.optimizer.zero_grad()
        
        out = self.prediction_model(inp)
        loss = self.loss_function(out, target)
        
        loss.backward()
        self.optimizer.step()

        if update_counter % self.update_target == 0 and update_counter > 0:
            self.target_model.load_state_dict(self.prediction_model.state_dict())
            torch.save(self.target_model.state_dict(), r"D:\Projects\Reinforcement Learning\LearningSnake\model\target_model.pth")

        return True

    def get_action(self, state):

        if np.random.random() > self.epsilon:
            # Get action from Q table
            state = torch.FloatTensor(state).view(1, -1)
            state = state.to(device)
            prediction = self.prediction_model(state)
            action = torch.argmax(prediction).item()

        else:
            # Select a random action
            action = np.random.randint(0, self.action_space_size)

        return action

    def decay_epsilon(self):
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.min_epsilon
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)