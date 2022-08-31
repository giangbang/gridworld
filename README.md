# gridworld
gridworld environment for RL 

## Installation

```
pip install git+https://github.com/giangbang/gridworld.git
```

## Code example

```python
import gridworld
import gym

env = gym.make('Gridworld-v0', plan=4, generate_goal=False, random_start=True)
while True:
  action = env.action_space.sample()
  _, _, done, _ = env.step(action)
  if done: break
```

