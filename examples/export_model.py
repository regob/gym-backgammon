import torch
import gymnasium as gym
from agents import LearningAgent

def export_onnx(net, input_names=["input"], output_names=["output"]):
    dummy_input = torch.randn(1, 198)
    torch.onnx.export(net.cpu(), dummy_input, "bkgm_net.onnx", verbose=True,
                      input_names=input_names, output_names=output_names)
    
        
if __name__ == "__main__":
    env = gym.make('gym_backgammon:backgammon-v0', render_mode="human")
    agent1 = LearningAgent(0, env.observation_space.shape, 256, debug=False)
    agent1.load_state()
    export_onnx(agent1.net)
