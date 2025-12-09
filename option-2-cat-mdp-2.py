import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Grid dimensions
grid_rows = 5
grid_cols = 5

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
action_up = 0
action_down = 1
action_left = 2
action_right = 3
actions = [action_up, action_down, action_left, action_right]
action_names = {0: 'AU', 1: 'AD', 2: 'AL', 3: 'AR'}
action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
num_actions = len(actions)

# Special locations
food_state = (4, 4)  
monster_states = [(0, 3), (4, 1)] 
forbidden_furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]
initial_state = (0, 0)

# Rewards
reward_default = -0.05
reward_food = 10.0
reward_monster = -8.0

# Dynamics probabilities
prob_intended = 0.70  
prob_right_turn = 0.12  
prob_left_turn = 0.12 
prob_stay = 0.06

# Discount factor
gamma = 0.925


def is_valid_state(state: Tuple[int, int]) -> bool:
    """Check if a state is valid (within bounds and not forbidden furniture)"""
    row, col = state
    # Check if within grid bounds
    if row < 0 or row >= grid_rows or col < 0 or col >= grid_cols:
        return False
    # Check if it's not forbidden furniture
    if state in forbidden_furniture:
        return False
    return True


def get_next_state(state: Tuple[int, int], action: int) -> Tuple[int, int]:
    
    row, col = state
    
    if action == action_up:
        next_state = (row - 1, col)
    elif action == action_down:
        next_state = (row + 1, col)
    elif action == action_left:
        next_state = (row, col - 1)
    elif action == action_right:
        next_state = (row, col + 1)
    else:
        next_state = state
    
    # If next state is invalid (wall or furniture), stay in current state
    if not is_valid_state(next_state):
        return state
    
    return next_state


def get_perpendicular_actions(action: int) -> Tuple[int, int]:
    perpendicular = {
        action_up: (action_right, action_left),
        action_down: (action_left, action_right),
        action_left: (action_up, action_down),
        action_right: (action_down, action_up)
    }
    return perpendicular[action]


class CatMonstersEnvironment:
    
    def __init__(self):
        self.state = initial_state
        self.done = False
    
    def reset(self):
        self.state = initial_state
        self.done = False
        return self.state
    
    def step(self, action: int):
        """
        Execute action in environment
        Args:
            action: int (0=Up, 1=Down, 2=Left, 3=Right)
        Returns: (next_state, reward, done)
        """
        if self.done:
            return self.state, 0.0, True
        
        current_state = self.state
        
        # Terminal state check
        if current_state == food_state:
            return current_state, 0.0, True
        
        # Determine actual action based on stochastic dynamics
        rand = np.random.random()
        
        if rand < prob_intended:
            actual_action = action
        elif rand < prob_intended + prob_right_turn:
            right_action, _ = get_perpendicular_actions(action)
            actual_action = right_action
        elif rand < prob_intended + prob_right_turn + prob_left_turn:
            _, left_action = get_perpendicular_actions(action)
            actual_action = left_action
        else:  # prob_stay
            actual_action = None  # Stay in place
        
        # Get next position
        if actual_action is None:
            next_state = current_state
        else:
            next_state = get_next_state(current_state, actual_action)
        
        # Calculate reward based on next state
        # Only food_state is terminal, monsters are not terminal
        if next_state == food_state:
            reward = reward_food
            done = True
        elif next_state in monster_states:
            reward = reward_monster
            done = False  # Monster states are NOT terminal
        else:
            reward = reward_default
            done = False
        
        self.state = next_state
        self.done = done
        
        return next_state, reward, done
    
    def get_state_features(self, state: Tuple[int, int]) -> np.ndarray:
        """Convert state to feature vector for function approximation"""
        r, c = state
        # One-hot encoding for 5x5 grid = 25 features
        features = np.zeros(grid_rows * grid_cols)
        features[r * grid_rows + c] = 1.0
        return features


class PolicyNetwork(nn.Module):
    """Neural network policy parameterization pi(a|s, theta) with 2 hidden layers"""
    
    def __init__(self, state_dim, num_actions, neurons_per_layer=(16, 16), learning_rate=5e-4):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.temperature = 1.0
        
        # Build 2-hidden-layer network
        layers = []
        last = state_dim
        for h in neurons_per_layer:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, num_actions))
        self.net = nn.Sequential(*layers)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def get_action_probs(self, state_features):
        """Get action probabilities from state features
        Args:
            state_features: numpy array
        Returns:
            action_probs: numpy array of probabilities
        """
        if isinstance(state_features, np.ndarray):
            state_tensor = torch.from_numpy(state_features.astype(np.float32))
        else:
            state_tensor = state_features
        
        with torch.no_grad():
            logits = self.forward(state_tensor)
            probs = F.softmax(logits / self.temperature, dim=-1)
            return probs.numpy()
    
    def select_action(self, state_features):
        """Sample action from policy pi(a|s, theta)
        Returns: action index (0=Up, 1=Down, 2=Left, 3=Right)"""
        action_probs = self.get_action_probs(state_features)
        action_idx = np.random.choice(self.num_actions, p=action_probs)
        return action_idx
    
    def get_action_and_log_prob(self, state_features):
        """
        Sample action from policy and return both action and log probability
        Args:
            state_features: numpy array
        Returns: (action_idx, log_prob_tensor)
        """
        if isinstance(state_features, np.ndarray):
            state_tensor = torch.from_numpy(state_features.astype(np.float32))
        else:
            state_tensor = state_features
        
        logits = self.forward(state_tensor)
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        return action_idx.item(), log_prob
    
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
    """Neural network value function approximation v̂(s, w) with 2 hidden layers"""
    
    def __init__(self, state_dim, neurons_per_layer=(16, 16), learning_rate=1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        
        # Build 2-hidden-layer network
        layers = []
        last = state_dim
        for h in neurons_per_layer:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: state -> value"""
        return self.net(x).squeeze(-1)
    
    def predict(self, state_features):
        """Predict value for a state
        Args:
            state_features: numpy array
        """
        if isinstance(state_features, np.ndarray):
            state_tensor = torch.from_numpy(state_features.astype(np.float32))
        else:
            state_tensor = state_features
        
        with torch.no_grad():
            value = self.forward(state_tensor)
            return value.item()
    
    def update(self, states, targets):
        """
        Update value function using batch of states and target values
        Args:
            states: List of state features (numpy arrays)
            targets: List of target values (returns G_t)
        Returns:
            loss: MSE loss value
        """
        self.optimizer.zero_grad()
        
        # Convert to tensors
        states_tensor = torch.from_numpy(np.array(states, dtype=np.float32))
        targets_tensor = torch.from_numpy(np.array(targets, dtype=np.float32))
        
        # Predict values
        predictions = self.forward(states_tensor)
        
        # MSE loss
        loss = F.mse_loss(predictions, targets_tensor)
        
        # Backpropagate and update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()


def generate_episode(env, policy):
    """
    Generate an episode following policy pi(a|s, theta)
    Returns: (episode, log_probs)
        episode: list of (state, action_idx, reward) tuples
        log_probs: list of log probability tensors for REINFORCE
    """
    episode = []
    log_probs = []
    state = env.reset()
    done = False
    
    while not done:
        state_features = env.get_state_features(state)
        action_idx, log_prob = policy.get_action_and_log_prob(state_features)
        
        next_state, reward, done = env.step(action_idx)
        
        episode.append((state, action_idx, reward))
        log_probs.append(log_prob)
        state = next_state
    
    return episode, log_probs


def reinforce_with_baseline(env, num_episodes=1000, alpha_theta=1e-3, alpha_w=1e-2, 
                            gamma=gamma, verbose=True):
    """
    REINFORCE with Baseline algorithm 
    
    Args:
        env: Environment
        num_episodes: Number of episodes to train
        alpha_theta: Step size for policy parameters
        alpha_w: Step size for value function weights
        gamma: Discount factor
        verbose: Whether to print progress
    
    Returns:
        policy: Trained policy network
        value_function: Trained value network
        episode_rewards: List of total rewards per episode
        value_losses: List of MSE losses for value function per episode
    """
    
    # Initialize policy pi(a|s, theta) and value function v(s, w) as neural networks
    state_dim = grid_rows * grid_cols
    policy = PolicyNetwork(state_dim, num_actions, neurons_per_layer=(16, 16), learning_rate=alpha_theta)
    value_function = ValueNetwork(state_dim, neurons_per_layer=(16, 16), learning_rate=alpha_w)
    
    episode_rewards = []
    value_losses = []
    
    for episode_num in range(num_episodes):
        episode, log_probs = generate_episode(env, policy)
        T = len(episode)
        
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
            state_features_t = env.get_state_features(state_t)
            
            G_t = 0.0
            for i in range(t, T):
                _, _, reward_i = episode[i]  # reward_i = R_{i+1}
                G_t += (gamma ** (i - t)) * reward_i
            
            # Compute baseline and advantage
            baseline = value_function.predict(state_features_t)
            advantage = G_t - baseline
            
            states.append(state_features_t)
            returns.append(G_t)
            advantages.append(advantage)
        
        # Update value function with batch
        mse_loss = value_function.update(states, returns)
        value_losses.append(mse_loss)
        
        # Normalize advantages to reduce variance
        advantages = np.array(advantages)
        if len(advantages) > 1 and np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        advantages = advantages.tolist()
        
        # Update policy with REINFORCE
        policy.update_with_reinforce(log_probs, advantages)
        
        # Print progress
        if verbose and (episode_num + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode_num + 1}/{num_episodes}, "
                  f"Avg Discounted Return (last 100): {avg_reward:.2f}, "
                  f"Current Episode Discounted Return: {G_0:.2f}")
    
    return policy, value_function, episode_rewards, value_losses


def evaluate_policy(env, policy, num_episodes=100, gamma=gamma):
    """Evaluate a learned policy using discounted returns"""
    discounted_returns = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        discounted_return = 0.0
        steps = 0
        
        policy.eval()  # Set policy to evaluation mode
        with torch.no_grad():
            while not done and steps < 1000:  # Max steps to prevent infinite loops
                state_features = env.get_state_features(state)
                action_idx = policy.select_action(state_features)  # 0=Up, 1=Down, 2=Left, 3=Right
                
                next_state, reward, done = env.step(action_idx)
                discounted_return += (gamma ** steps) * reward
                state = next_state
                steps += 1
        
        policy.train()  # Set policy back to training mode
        
        discounted_returns.append(discounted_return)
        episode_lengths.append(steps)
    
    return {
        'mean_reward': np.mean(discounted_returns),
        'std_reward': np.std(discounted_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }


def plot_learning_curve(episode_rewards, window=100):
    # Raw episode discounted returns
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Returns')
    plt.xlabel('Episode')
    plt.ylabel('Discounted Return')
    plt.title('Episode Discounted Returns')
    plt.grid(True, alpha=0.3)
    plt.locator_params(axis='y', nbins=12)  # More y-axis tick marks
    plt.legend()
    plt.tight_layout()
    plt.savefig('reinforce_baseline_learning_curve_cat_mdp_5.png', dpi=150)
    print("Learning curve saved as 'reinforce_baseline_learning_curve_cat_mdp_5.png'")
    plt.show()
    
    # Moving average
    plt.figure(figsize=(10, 6))
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window})')
        
        # Find first point where moving average exceeds 2.5
        threshold = 2.5
        indices_above = np.where(moving_avg > threshold)[0]
        if len(indices_above) > 0:
            first_idx = indices_above[0]
            first_episode = first_idx  # Episode number (0-indexed in the moving avg array)
            first_value = moving_avg[first_idx]
            
            # Mark the point
            plt.plot(first_episode, first_value, 'go', markersize=12, 
                    label=f'First > {threshold} at episode {first_episode}', zorder=5)
            
            # Add annotation
            # plt.annotate(f'Episode {first_episode}\nReturn: {first_value:.2f}',
            #             xy=(first_episode, first_value),
            #             xytext=(first_episode + 200, first_value + 0.3),
            #             fontsize=10,
            #             bbox=dict(alpha=0.7))
            
            # Add horizontal reference line at threshold
            plt.axhline(y=threshold, color='green', linestyle='--', linewidth=1, 
                       alpha=0.5, label=f'Threshold: {threshold}')
        
        plt.xlabel('Episode')
        plt.ylabel(f'Average Discounted Return')
        plt.title('Moving Average of Discounted Returns')
        plt.grid(True, alpha=0.3)
        plt.locator_params(axis='y', nbins=12)  # More y-axis tick marks
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('reinforce_baseline_learning_curve_cat_mdp_moving_avg_5.png', dpi=150)
        print("Moving average plot saved as 'reinforce_baseline_learning_curve_cat_mdp_moving_avg_5.png'")
        plt.show()


def plot_value_function_mse(value_losses, window=100):
    """Plot the MSE of value function predictions vs actual returns"""
    
    # Raw MSE per episode
    plt.figure(figsize=(10, 6))
    plt.plot(value_losses, alpha=0.3, color='orange', label='Value Function MSE')
    plt.xlabel('Episode')
    plt.ylabel('MSE Loss')
    plt.title('Value Function MSE (v̂ vs G)')
    plt.grid(True, alpha=0.3)
    plt.locator_params(axis='y', nbins=12)  # More y-axis tick marks
    plt.legend()
    plt.tight_layout()
    plt.savefig('cat_mdp_value_function_mse_5.png', dpi=150)
    print("Value function MSE plot saved as 'cat_mdp_value_function_mse_5.png'")
    plt.show()
    
    # Moving average of MSE
    plt.figure(figsize=(10, 6))
    if len(value_losses) >= window:
        moving_avg = np.convolve(value_losses, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, color='darkred', linewidth=2, label=f'Moving Average (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average MSE Loss')
        plt.title('Moving Average of Value Function MSE')
        plt.grid(True, alpha=0.3)
        plt.locator_params(axis='y', nbins=12)  # More y-axis tick marks
        plt.legend()
        plt.tight_layout()
        plt.savefig('cat_mdp_value_function_mse_moving_avg_5.png', dpi=150)
        print("Moving average MSE plot saved as 'cat_mdp_value_function_mse_moving_avg_5.png'")
        plt.show()


def visualize_policy(env, policy):
    policy_grid = [['' for _ in range(grid_cols)] for _ in range(grid_rows)]
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            state = (r, c)
            state_features = env.get_state_features(state)
            action_probs = policy.get_action_probs(state_features)
            best_action_idx = np.argmax(action_probs)
            
            # Mark special states
            if state == food_state:
                policy_grid[r][c] = 'G'
            elif state in monster_states:
                policy_grid[r][c] = 'M'
            elif state in forbidden_furniture:
                policy_grid[r][c] = 'X'
            elif state == initial_state:
                policy_grid[r][c] = f'C{action_symbols[best_action_idx]}'
            else:
                policy_grid[r][c] = action_symbols[best_action_idx]
    
    for r in range(grid_rows):
        row_str = ' '.join(f'{policy_grid[r][c]:^4}' for c in range(grid_cols))
        print(row_str)
        # if r < grid_rows - 1:
        #     print('-' * 70)

    print("Policy:")
    for r in range(grid_rows):
        for c in range(grid_cols):
            state = (r, c)
            if state in forbidden_furniture:
                print("  X  ", end=" ")
            elif state == food_state:
                print("  G  ", end=" ")
            elif state in monster_states:
                print("  M  ", end=" ")
            else:
                state_features = env.get_state_features(state)
                action_probs = policy.get_action_probs(state_features)
                best_action_idx = np.argmax(action_probs)
                print(f" {action_names[best_action_idx]}  ", end=" ")
        print()
    print()


def main():
    env = CatMonstersEnvironment()
    # Training parameters
    num_episodes = 3000
    alpha_theta = 5e-4  # Step size for policy parameters (using Adam optimizer)
    alpha_w = 1e-3      # Step size for value function (using Adam optimizer)
    
    print(f"\nTraining Parameters:")
    print(f"  Number of Episodes: {num_episodes}")
    print(f"  Policy Learning Rate: {alpha_theta}")
    print(f"  Value Learning Rate: {alpha_w}")
    print(f"  Discount Factor: {gamma}")
    print(f"  Policy Network: 2 hidden layers (16, 16)")
    print(f"  Value Network: 2 hidden layers (16, 16)")
    print(f"  Using advantage normalization for variance reduction")
    print("\nStarting training\n")
    
    policy, value_function, episode_rewards, value_losses = reinforce_with_baseline(
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
    
    visualize_policy(env, policy)
    
    plot_learning_curve(episode_rewards, window=200)
    plot_value_function_mse(value_losses, window=200)


if __name__ == "__main__":
    main()

