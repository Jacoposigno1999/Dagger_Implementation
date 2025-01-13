from Agent import Agent
from Utils import find_last_episode


# Directory containing saved models
model_directory = r'C:\Users\39388\Desktop\Dagger_project\Models'

# Check the last saved model
Last_trained_model = find_last_episode(model_directory)

print(Last_trained_model)
model_path = r"C:\Users\39388\Desktop\Dagger_project\Models\model_{}.pth".format(Last_trained_model)
agent = Agent(name="test_agent", input_num=96 * 96, output_num=3)
agent.test_model(model_path=model_path, n_episodes=1)