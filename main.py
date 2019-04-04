from LearningAlgorithms.q_learner import Q_Learner
import gym
import numpy as np

def play(agent, training=False, verbose=True):
  env = gym.make('MsPacman-ram-v0')
  state = env.reset() # initialize the environment

  score = 0
  done = False
  rounds = 0
  while(not done):
    action = agent.best_action(state, training=training)

    old_state = state
    state, reward, done, _ = env.step(action)

    if training:
      agent.update_parameters(old_state, state, action, reward)

    score += reward
    rounds += 1

  if verbose:
    print("--> Game Over. Rounds: {} | Score: {}".format(rounds, score))

  return rounds, score


class My_Q_Learner(Q_Learner):
  def features(self, state, action):
    return state


def main():

  # initialization
  actions_arr = np.arange(9)
  d = 128
  learning_rate = 0.01
  epsilon = 0.9
  discount_factor = 0.8

  agent = My_Q_Learner(actions_arr, d, learning_rate, epsilon, discount_factor)

  # Training loop
  num_matches = 10
  for i in range(num_matches):
    play(agent, training=True, verbose=True)
  
  print("Training completed.\n")

  ###

  print("Expert agent playing...")
  score = play(agent)
  print("Finished! Score: {}".format(score))

  print(agent.theta)

if __name__ == '__main__':
  main()