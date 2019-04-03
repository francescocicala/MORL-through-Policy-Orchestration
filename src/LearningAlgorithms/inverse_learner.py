import numpy as np
import abc
import gym

from q_learner import Q_Learner


class My_Q_Learner(Q_Learner):
  def features(self, state, action):
    return state



class Inverse_Learner(abc.ABC):
  def __init__(self, dim, expert_trajectories, discount_factor):
    self.dim = dim
    self.discount_factor = discount_factor
    self.f_expect_expert = self.compute_feat_expectations(expert_trajectories)
    self.feat_expectations = []
    self.feat_exp_proj = []
    self.w = np.random.normal(0, 0.1, dim)
    self.t = []


  @abc.abstractmethod
  def features(self, old_state, action, state):
    pass


  def initialize_parameters(self, actions_arr, d, learning_rate, epsilon, disc_factor, num_matches=10):

    if (len(self.feat_expectations) == 0 and len(self.feat_exp_proj) == 0):
      agent = My_Q_Learner(actions_arr, d, learning_rate, epsilon, disc_factor)
      agent = self.train_agent(agent, self.w, num_matches)

      trajectory = self.produce_trajectory(agent)
      self.feat_expectations.append(self.compute_feat_expectations(trajectory))
      self.feat_exp_proj.append(self.feat_expectations[0])
      self.w = self.f_expect_expert - self.feat_expectations[0]
      self.t.append(np.linalg.norm(self.w))

    else:
      raise Exception("self.feat_expectations and self.feat_exp_proj should be empty!")


  def update_parameters(self, actions_arr, d, learning_rate, epsilon, disc_factor, num_matches=10):
    agent = My_Q_Learner(actions_arr, d, learning_rate, epsilon, disc_factor)
    agent = self.train_agent(agent, self.w, num_matches)
    trajectory = self.produce_trajectory(agent)

    mu1 = self.compute_feat_expectations(trajectory)
    self.feat_expectations.append(mu1)

    mu2_ = self.feat_exp_proj[-1]
    mu2 = self.feat_expectations[-2]
    muE = self.f_expect_expert
    mu1_ = mu2_ + (mu1 - mu2_).dot(muE - mu2_) * (mu1 - mu2_) / (mu1 - mu2_).dot(mu1 - mu2_) 
    self.feat_exp_proj.append(mu1_)

    self.w = self.f_expect_expert - mu1_
    self.t.append(np.linalg.norm(self.w))


    

  def compute_feat_expectations(self, trajectory):
    feat_expect = np.zeros(self.dim)
    for i in range(2, len(trajectory), 2):
      feat_expect = feat_expect + self.discount_factor ** ((i-2)/2) * self.features(trajectory[i-2], trajectory[i-1], trajectory[i])

    return feat_expect

  def train_agent(self, agent, weights=None, num_matches=10):
    env = gym.make('MsPacman-ram-v0')
    state = env.reset() # initialize the environment

    if type(weights) == type(None):
      weights = self.w

    for i in range(num_matches):

      done = False
      while(not done):
        action = agent.best_action(state, training=True)

        old_state = state
        state, _, done, _ = env.step(action)

        reward = weights.dot(self.features(old_state, action, state))
        agent.update_parameters(old_state, state, action, reward)

    return agent



  def produce_trajectory(self, agent):
    env = gym.make('MsPacman-ram-v0')
    state = env.reset() # initialize the environment
    trajectory = [state]

    done = False
    while(not done):
      action = agent.best_action(state, training=False)
      state, _, done, _ = env.step(action)
      trajectory.append(action)
      trajectory.append(state)

    return trajectory



class My_Inverse_Learner(Inverse_Learner):
  def features(self, state, action):
    return state
    

if __name__ == '__main__':
  expert_traj = [np.random.uniform(-1, 1, 128), 
                 2, 
                 np.random.uniform(-1, 1, 128)]
  irl = Inverse_Learner(128, expert_traj, 0.95)
  
  # initialization
  actions_arr = np.arange(9)
  d = 128
  learning_rate = 0.01
  epsilon = 0.9
  discount_factor = 0.8

  irl.initialize_parameters(actions_arr, d, learning_rate, epsilon, discount_factor, num_matches=10, )
  
  for i in range(10):
    irl.update_parameters(actions_arr, d, learning_rate, epsilon, discount_factor, num_matches=10)
  print(irl.t)