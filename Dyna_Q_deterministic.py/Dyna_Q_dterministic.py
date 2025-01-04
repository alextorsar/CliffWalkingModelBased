import gymnasium as gym
import numpy as np

env = gym.make('CliffWalking-v0', is_slippery=False)
action_space = env.action_space
state_space = env.observation_space

q = np.zeros((state_space.n, action_space.n))
model = {}

def print_policy(policy):
    grid = [[' ' for _ in range(12)] for _ in range(4)]
    for state in range(state_space.n):
        row = state // 12
        col = state % 12
        if state == 36:
            grid[row][col] = 'G'
        elif state == 47:
            grid[row][col] = 'C'
        else:
            grid[row][col] = ['^', '>', 'v', '<'][policy[state]]
    for row in grid:
        print(row)
        
def epsilon_greedy(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(action_space.n)
    else:
        return np.argmax(Q[state, :])

alpha = 1
gamma = 0.9
epsilon = 0.1
num_episodes = 200
num_iterations = 20

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    while not done:
        action = epsilon_greedy(q,state,epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        if state not in model:
            model[state] = {}
        model[state][action] = (reward, next_state)
        q[state,action] = q[state,action] + alpha * (reward + (gamma * np.max(q[next_state,:])) - q[state,action])
        for _ in range(num_iterations):
            s = np.random.choice(list(model.keys()))
            a = np.random.choice(list(model[s].keys()))
            r, s_ = model[s][a]
            q[s,a] = q[s,a] + alpha * (r + (gamma * np.max(q[s_,:])) - q[s,a])
        if not done:
            state = next_state

policy = {}
for state in range(state_space.n):
    policy[state] = np.argmax(q[state,:])
    
print_policy(policy)

env = gym.make('CliffWalking-v0', is_slippery=False, render_mode='human')

test_episodes = 5
for episode in range(test_episodes):
    state, info = env.reset()
    done = False
    while not done:
        action = policy[state]
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state