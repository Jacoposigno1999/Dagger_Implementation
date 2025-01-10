import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym 


class Agent(nn.Module):
    def __init__(self, name='model', input_num=None, output_num=None):
        super(Agent, self).__init__()
        assert input_num is not None
        assert output_num is not None
        self.input_num = input_num
        self.output_num = output_num
        #self._build_net()
        self.conv1 = nn.Conv2d(1,4,5)
        self.norm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4,8,5)
        self.norm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8,16,6)
        self.norm3 = nn.BatchNorm2d(16)
        self.lin0  = nn.Linear(1024,768)#layer aggiunto da me 
        self.lin1  = nn.Linear(768,64)
        self.lin2  = nn.Linear(64,16)
        self.lin3  = nn.Linear(16,3)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.1)
        self.relu    = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.maxpool(x)
        x = self.relu(x)


        x = self.flatten(x)

        x = self.lin0(x)
        x = self.relu(x)


        x = self.lin1(x)
        x = self.relu(x)
       
        x = self.lin2(x)
        x = self.relu(x)

        x = self.lin3(x)
        x = self.relu(x)
        
        return x
    
    def train_model(self, x, y, n_epoch=100, batch=32):
        """Train the network"""

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)


        #x = torch.tensor(x, dtype=torch.float32)
        #y = torch.tensor(y, dtype=torch.float32)

        x = torch.stack(x)
        y = torch.stack(y)
        print(f'x shape: {x.shape}, y shape: {y.shape}')

        dataset = torch.utils.data.TensorDataset(x,y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle= True)

        #Training Loop
        for epoch in range(n_epoch):
            epoch_loss = 0.0
            self.train()
            n = 0 
            for batc_x, batch_y in dataloader:
                optimizer.zero_grad()
                prediction = self(batc_x)
                loss = loss_fn(prediction, batch_y)
                print(f'prediction{prediction[1]} true value: {batch_y[1]}') if n == 50 else None 
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1 

            print(f"Epoch {epoch + 1}/{n_epoch}, Loss: {epoch_loss:.4f}, Avg. Epoc Loss:{epoch_loss/n}")

        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.reshape(1, 1, 96, 96)
            predictions = self(x)
        
        return predictions.numpy()
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)



    def test_model(self, model_path=None, n_episodes=10, render=True):

        if model_path:
            self.load(model_path)  # Load the trained model
        self.eval()  # Set the model to evaluation mode

        # Initialize the environment
        env = gym.make("CarRacing-v3", render_mode="human" if render else None)

        total_rewards = []

        for episode in range(n_episodes):
            print(f"Starting episode {episode + 1}/{n_episodes}")

            # Reset the environment
            observation, info = env.reset()
            episode_reward = 0
            negative_reward = 0
            terminated, truncated = False, False

            while not (terminated or truncated):
                # Preprocess observation (convert to grayscale and normalize)
                gray_observation = torch.tensor(
                    (observation[..., :3] @ [0.2126, 0.7152, 0.0722]).astype("float32")
                ).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

                # Predict the action
                with torch.no_grad():
                    action = self.predict(gray_observation).reshape(3,)#.squeeze(0).numpy()

                # Step in the environment
                observation, reward, terminated, truncated, info = env.step(action)

                # Accumulate rewards
                episode_reward += reward
                
                if reward < 0:
                    negative_reward = negative_reward + reward

                if negative_reward < -20:
                    break 

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1} finished with reward: {episode_reward}")

        env.close()
        

        # Print summary statistics
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average Reward over {n_episodes} episodes: {avg_reward:.2f}")
        print(f"Total Rewards: {total_rewards}")

        return total_rewards