from LearningAlgorithms.q_learner import Q_Learner
import gym
import numpy as np

def play(agent, num_matches=10):
  rewards = []
  for i in range(num_matches):
    print("Match {}\n".format(i))
    env = gym.make('MsPacman-ram-v0')
    state = env.reset() # initialize the environment

    rewards.append(0)
    done = False
    while(not done):
      action = agent.best_action(state, training=False)
      state, reward, done, _ = env.step(action)
      rewards[i] += reward

  return np.mean(rewards)


def main():

  actions_arr = np.arange(9)
  d = 128
  learning_rate = 0.01
  epsilon = 0.9
  discount_factor = 0.8

  agent = Q_Learner(actions_arr, d, learning_rate, epsilon, discount_factor)
  
  print("Noob agent playing...\n")
  score = play(agent)
  print("Finished! Score: {}".format(score))

  # Training loop
  num_matches = 10
  for i in range(num_matches):
    env = gym.make('MsPacman-ram-v0')
    state = env.reset() # initialize the environment

    score = 0
    done = False
    while(not done):
      action = agent.best_action(state, training=True)

      old_state = state
      state, reward, done, _ = env.step(action)

      agent.update_parameters(old_state, state, action, reward)
      score += reward
      
    print("Match {} completed. Score: {}".format(i, score))
  
  print("Training completed.\n")

  ###

  print("Expert agent playing...")
  score = play(agent)
  print("Finished! Score: {}".format(score))

  print(agent.theta)

if __name__ == '__main__':
  main()