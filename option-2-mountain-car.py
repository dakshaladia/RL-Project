import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mountain Car Constants
min_position = -1.2
max_position = 0.5
min_velocity = -0.07
max_velocity = 0.07
max_steps = 2000

# Actions: -1=Reverse, 0=Neutral, 1=Forward
actions = [-1, 0, 1]
num_actions = len(actions)

# Discount factor
gamma = 1.0


def clip(value, min_val, max_val):
    """Clip value to be within [min_val, max_val]"""
    return max(min_val, min(value, max_val))


class MountainCarEnvironment:
    """Mountain Car MDP Environment"""
    
    def __init__(self):
        self.state = None
        self.done = False
        self.steps = 0
    
    def reset(self):
        """Reset environment to initial state"""
        # Initial position uniformly random from [-0.6, -0.4]
        x_0 = np.random.uniform(-0.6, -0.4)
        self.state = (x_0, 0.0)  # (position, velocity)
        self.done = False
        self.steps = 0
        return self.state
    
    def step(self, action: int):
        """
        Execute action in environment
        Args:
            action: int (-1=Reverse, 0=Neutral, 1=Forward)
        Returns: (next_state, reward, done)
        """
        if self.done:
            return self.state, 0.0, True
        
        x_t, v_t = self.state
        
        # Dynamics: deterministic
        # Step 1: Update velocity
        v_next = v_t + 0.001 * action - 0.0025 * np.cos(3 * x_t)
        
        # Step 2: Clip velocity BEFORE updating position
        v_next = clip(v_next, min_velocity, max_velocity)
        
        # Step 3: Update position using CLIPPED velocity
        x_next = x_t + v_next
        
        # Step 4: Clip position
        x_next = clip(x_next, min_position, max_position)
        
        # Step 5: Collision simulation - if at bounds, velocity resets to 0
        if x_next == min_position or x_next == max_position:
            v_next = 0.0
        
        self.state = (x_next, v_next)
        self.steps += 1
        
        # Terminal conditions
        if x_next == max_position:  # Reached goal
            self.done = True
            reward = 0.0  # Reward is 0 when terminating due to success
        elif self.steps >= max_steps:  # Timeout
            self.done = True
            reward = -1.0  # Still get -1 for this step
        else:
            reward = -1.0  # -1 at every timestep
            self.done = False
        
        return self.state, reward, self.done
    
    def get_state_features(self, state: Tuple[float, float]) -> np.ndarray:
        """Convert state to feature vector for function approximation
        Uses polynomial features: [1, x, v, x^2, v^2, x*v, x^3, v^3, ...]
        """
        x, v = state
        
        # Normalize position and velocity to roughly [-1, 1] range for better numerical stability
        x_norm = (x - min_position) / (max_position - min_position) * 2 - 1  
        v_norm = v / max_velocity  

        features = np.array([
            1.0,                    
            x_norm,                 
            v_norm,               
            x_norm ** 2,           
            v_norm ** 2,          
            x_norm * v_norm,      
            x_norm ** 3,           
            v_norm ** 3,           
            (x_norm ** 2) * v_norm, 
            x_norm * (v_norm ** 2), 
            np.sin(np.pi * x_norm), 
            np.cos(np.pi * x_norm), 
            np.sin(2 * np.pi * x_norm),  
            np.cos(2 * np.pi * x_norm),  
            np.sin(np.pi * v_norm), 
            np.cos(np.pi * v_norm), 
        ])
        
        return features


class PolicyNetwork(nn.Module):
    
    def __init__(self, neurons_per_layer=(32, 32), learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.temperature = 1.0
        
        layers = []
        n_inputs = 2   # State is [x, v]
        n_outputs = 3  # Three actions: -1, 0, 1
        
        last = n_inputs
        for h in neurons_per_layer:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, n_outputs))
        self.net = nn.Sequential(*layers)
        
        # Normalization constants
        self.pos_mid = 0.5 * (min_position + max_position)
        self.pos_half = 0.5 * (max_position - min_position)
        self.vel_mid = 0.5 * (min_velocity + max_velocity)
        self.vel_half = 0.5 * (max_velocity - min_velocity)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize state to [-1, 1]"""
        pos = x[..., 0]
        vel = x[..., 1]
        pos_n = (pos - self.pos_mid) / self.pos_half
        vel_n = (vel - self.vel_mid) / self.vel_half
        return torch.stack([pos_n, vel_n], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self._normalize(x)
        return self.net(x_n)
    
    def get_action_and_log_prob(self, state):
        """
        Sample action from policy and return both action and log probability
        Args:
            state: tuple (x, v) or numpy array [x, v]
        Returns: (action_idx, log_prob_tensor)
        """
        # Convert tuple or list to numpy array
        if isinstance(state, (tuple, list)):
            state = np.array(state, dtype=np.float32)
        elif isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        
        state_tensor = torch.from_numpy(state)
        logits = self.forward(state_tensor)
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        return action_idx.item(), log_prob
    
    def select_action(self, state):
        """Sample action from policy (for compatibility)
        Args:
            state: tuple (x, v) or numpy array [x, v]
        """
        action_idx, _ = self.get_action_and_log_prob(state)
        return action_idx
    
    def update_with_reinforce(self, log_probs, advantages):
        """
        Update policy using REINFORCE with baseline
        Args:
            log_probs: List of log probability tensors from episode
            advantages: List of advantage values (G_t - baseline)
        """
        self.optimizer.zero_grad()
        
        policy_loss = 0
        for log_prob, advantage in zip(log_probs, advantages):
            # Clip advantage to prevent exploding gradients
            clipped_advantage = np.clip(advantage, -10, 10)
            policy_loss += -log_prob * clipped_advantage
        
        # Backpropagate and update
        policy_loss.backward()
        
        # Clip gradients to prevent exploding
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()


class ValueNetwork(nn.Module):
    """Neural network value function approximation v̂(s, w) using PyTorch"""
    
    def __init__(self, neurons_per_layer=(32, 32), learning_rate=1e-2):
        super().__init__()
        self.learning_rate = learning_rate
        
        layers = []
        n_inputs = 2  
        n_outputs = 1
        
        last = n_inputs
        for h in neurons_per_layer:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, n_outputs))
        self.net = nn.Sequential(*layers)
        
        self.pos_mid = 0.5 * (min_position + max_position)
        self.pos_half = 0.5 * (max_position - min_position)
        self.vel_mid = 0.5 * (min_velocity + max_velocity)
        self.vel_half = 0.5 * (max_velocity - min_velocity)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize state to [-1, 1]"""
        pos = x[..., 0]
        vel = x[..., 1]
        pos_n = (pos - self.pos_mid) / self.pos_half
        vel_n = (vel - self.vel_mid) / self.vel_half
        return torch.stack([pos_n, vel_n], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: state -> value"""
        x_n = self._normalize(x)
        return self.net(x_n).squeeze(-1)
    
    def predict(self, state) -> float:
        """Predict value for a state
        Args:
            state: tuple (x, v) or numpy array [x, v]
        """
        # Convert tuple or list to numpy array
        if isinstance(state, (tuple, list)):
            state = np.array(state, dtype=np.float32)
        elif isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state)
            value = self.forward(state_tensor)
            return value.item()
    
    def update(self, states, targets):
        """
        Update value function using batch of states and target values
        Args:
            states: List of states (tuples or numpy arrays)
            targets: List of target values (returns G_t)
        """
        self.optimizer.zero_grad()
        
        # Convert states to numpy array
        states_array = []
        for s in states:
            if isinstance(s, (tuple, list)):
                states_array.append(list(s))
            else:
                states_array.append(s)
        
        # Convert to tensors
        states_tensor = torch.from_numpy(np.array(states_array, dtype=np.float32))
        targets_tensor = torch.from_numpy(np.array(targets, dtype=np.float32))
        
        # Predict values
        predictions = self.forward(states_tensor)
        
        # MSE loss
        loss = F.mse_loss(predictions, targets_tensor)
        
        # Backpropagate and update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()


def generate_episode(env, policy):
    """
    Generate an episode following policy pi
    Returns: (episode, log_probs, final_state, reached_goal)
        episode: list of (state, action_idx, reward) tuples
        log_probs: list of log probability tensors for REINFORCE
        final_state: the final state after the last action
        reached_goal: whether the episode reached the goal
    """
    episode = []
    log_probs = []
    state = env.reset()
    done = False
    
    while not done:
        action_idx, log_prob = policy.get_action_and_log_prob(state)
        action = actions[action_idx]  
        
        next_state, reward, done = env.step(action)
        
        episode.append((state, action_idx, reward))
        log_probs.append(log_prob)
        state = next_state
    
    reached_goal = (state[0] == max_position)
    final_state = state
    
    return episode, log_probs, final_state, reached_goal


def reinforce_with_baseline(env, num_episodes=1000, alpha_theta=1e-3, alpha_w=1e-2, 
                            gamma=gamma, verbose=True):
    """
    REINFORCE with Baseline algorithm
    
    Args:
        env: Environment
        num_episodes: Number of episodes to train
        alpha_theta: Step size for policy parameters theta
        alpha_w: Step size for value function weights
        gamma: Discount factor
        verbose: Whether to print progress
    
    Returns:
        policy: Trained policy network
        value_function: Trained value network
        episode_rewards: List of total rewards per episode
    """
    
    # Initialize policy pi(a|s, theta) and value function v(s, w) as neural networks
    policy = PolicyNetwork(neurons_per_layer=(32, 32), learning_rate=alpha_theta)
    value_function = ValueNetwork(neurons_per_layer=(32, 32), learning_rate=alpha_w)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode_num in range(num_episodes):
      
        episode, log_probs, _, reached_goal = generate_episode(env, policy)
        T = len(episode)
        episode_lengths.append(T)
        
        # Track success
        if reached_goal:
            success_count += 1
        
        G_0 = 0.0
        for t in range(T):
            _, _, reward_t = episode[t]  
            G_0 += (gamma ** t) * reward_t
        episode_rewards.append(G_0)
        
        # Compute returns and advantages for all timesteps
        states = []
        returns = []
        advantages = []
        
        for t in range(T):
            state_t, action_t, reward_t = episode[t] 
            
            G_t = 0.0
            for i in range(t, T):
                _, _, reward_i = episode[i]  
                G_t += (gamma ** (i - t)) * reward_i
            
            # Compute baseline and advantage
            baseline = value_function.predict(state_t)
            advantage = G_t - baseline
            
            states.append(state_t)
            returns.append(G_t)
            advantages.append(advantage)
        
        # Update value function with batch
        value_function.update(states, returns)
        
        # Update policy with REINFORCE
        policy.update_with_reinforce(log_probs, advantages)
        
        if verbose and (episode_num + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            recent_lengths = episode_lengths[-100:]
            recent_success = sum(1 for i in range(max(0, len(episode_lengths) - 100), len(episode_lengths)) 
                               if episode_lengths[i] < max_steps)
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            success_rate = recent_success / min(100, episode_num + 1)
            print(f"Episode {episode_num + 1}/{num_episodes}, "
                  f"Avg Return: {avg_reward:.1f}, "
                  f"Avg Length: {avg_length:.0f}, "
                  f"Success Rate: {success_rate:.1%}, "
                  f"Current Length: {T}")
    
    return policy, value_function, episode_rewards


def evaluate_policy(env, policy, num_episodes=100, gamma=0.99):
    """Evaluate a learned policy using discounted returns"""
    discounted_returns = []
    episode_lengths = []
    success_count = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        discounted_return = 0.0
        steps = 0
        reached_goal = False
        
        policy.eval()  # Set policy to evaluation mode
        with torch.no_grad():
            while not done:
                action_idx = policy.select_action(state)
                action = actions[action_idx]
                
                next_state, reward, done = env.step(action)
                discounted_return += (gamma ** steps) * reward
                state = next_state
                steps += 1
                
                if next_state[0] == max_position:
                    reached_goal = True
        
        policy.train()  # Set policy back to training mode
        
        if reached_goal:
            success_count += 1
        
        discounted_returns.append(discounted_return)
        episode_lengths.append(steps)
    
    return {
        'mean_reward': np.mean(discounted_returns),
        'std_reward': np.std(discounted_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes
    }


def plot_learning_curve(episode_rewards, window=100):

    # Raw episode discounted returns
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Returns')
    plt.xlabel('Episode')
    plt.ylabel('Discounted Return')
    plt.title('Episode Discounted Returns')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mountain_car_reinforce_baseline_learning_curve_3.png', dpi=150)
    print("Learning curve saved as 'mountain_car_reinforce_baseline_learning_curve_3.png'")
    plt.show()
    
    # Moving average
    plt.figure(figsize=(10, 6))
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window})')
        plt.xlabel('Episode')
        plt.ylabel(f'Average Discounted Return')
        plt.title('Moving Average of Discounted Returns')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('mountain_car_reinforce_baseline_learning_curve_moving_avg_3.png', dpi=150)
        print("Moving average plot saved as 'mountain_car_reinforce_baseline_learning_curve_moving_avg_3.png'")
        plt.show()


def main():
    env = MountainCarEnvironment()
    
    # Training parameters
    num_episodes = 3000
    alpha_theta = 3e-4  # Learning rate for policy network (Adam optimizer)
    alpha_w = 1e-3      # Learning rate for value network (Adam optimizer)
    
    print(f"\nTraining Parameters:")
    print(f"  Number of Episodes: {num_episodes}")
    print(f"  Policy Learning Rate: {alpha_theta}")
    print(f"  Value Learning Rate: {alpha_w}")
    print(f"  Discount Factor: {gamma}")
    print("\nStarting training\n")
    
    # Train using REINFORCE with Baseline
    policy, value_function, episode_rewards = reinforce_with_baseline(
        env=env,
        num_episodes=num_episodes,
        alpha_theta=alpha_theta,
        alpha_w=alpha_w,
        gamma=gamma,
        verbose=True
    )
    
    print("\nEvaluating learned policy:")
    eval_results = evaluate_policy(env, policy, num_episodes=100)
    
    print(f"\nEvaluation Results (100 episodes):")
    print(f"  Mean Discounted Return: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
    print(f"  Success Rate (reached goal): {eval_results['success_rate']:.1%}")
    
    plot_learning_curve(episode_rewards, window=100)


if __name__ == "__main__":
    main()

