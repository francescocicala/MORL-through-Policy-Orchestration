import numpy as np
import abc
import gym

###################
#### Utilities ####
###################

def L2_normalization(vec):
  return vec / np.linalg.norm(vec)

###################
#### Q_Learner ####
###################

class Q_Learner(abc.ABC):
  """
  Q-learning agent for an environment where the set of possibile actions
  is finite.
  """
  def __init__(self, actions_arr, d, learning_rate, epsilon, discount_factor, theta=None):
    self.actions_arr = actions_arr
    self.d = d # number of features taken into account in the Q-function
    self.learning_rate = learning_rate # learning rate
    self.epsilon = epsilon # we use an epsilon-greedy strategy
    self.discount_factor = discount_factor # discounting factor

    if theta is None:
      self.theta = L2_normalization(np.random.normal(0, 0.1, d))

    elif len(theta) == d:
      self.theta = L2_normalization(theta)

    else:
      raise ValueError("theta length ({}) is different from d ({})".format(len(self.theta), d))


  @abc.abstractmethod
  def features(self, state, action):
    pass


  def Q(self, state, action):
    return self.features(state, action).dot(self.theta)


  def best_action(self, state, training=False):
    u = np.random.uniform()
    if u > self.epsilon and training:
      return self.actions_arr[np.random.randint(len(self.actions_arr))]
    else:
      table = np.asarray([self.Q(state, action) for action in self.actions_arr])
      return self.actions_arr[np.argmax(table)]


  def difference(self, old_state, new_state, action, reward):
    return reward + self.discount_factor * self.best_action(new_state) - self.Q(old_state, action)


  def update_parameters(self, old_state, new_state, action, reward):
    diff = self.difference(old_state, new_state, action, reward)
    self.theta += self.learning_rate * diff * self.features(old_state, action)
    self.theta = L2_normalization(self.theta) # some noise is added in this step 


#########################
#### Inverse_Learner ####
#########################

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


  @abc.abstractmethod
  def environment(self):
    pass


  @abc.abstractmethod
  def imitator(self):
    pass


  def initialize_parameters(self, num_matches=10):
    if (len(self.feat_expectations) == 0 and len(self.feat_exp_proj) == 0):
      agent = self.imitator()
      agent = self.train_agent(agent, self.w, num_matches)

      trajectory = self.produce_trajectory(agent)
      self.feat_expectations.append(self.compute_feat_expectations(trajectory))
      self.feat_exp_proj.append(self.feat_expectations[0])
      self.w = self.f_expect_expert - self.feat_expectations[0]
      self.t.append(np.linalg.norm(self.w))
    else:
      raise Exception("self.feat_expectations and self.feat_exp_proj should be empty!")


  #def update_parameters(self, actions_arr, d, learning_rate, epsilon, disc_factor, num_matches=10):
  def update_parameters(self, num_matches=10):
 
    agent = self.imitator()
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


  def compute_feat_expectations(self, trajectories):
    total_feat = np.zeros(self.dim)
    for trajectory in trajectories:
      feat_expect = np.zeros(self.dim)
      for i in range(2, len(trajectory), 2):
        feat_expect = feat_expect + self.discount_factor ** ((i-2)/2) * self.features(trajectory[i-2], trajectory[i-1], trajectory[i])
      total_feat = total_feat + feat_expect
    return total_feat


  def train_agent(self, agent, weights=None, num_matches=10):
    env = self.environment()
    if weights is None:
      weights = self.w

    for i in range(num_matches):
      state = env.reset() # initialize the environment
      done = False
      while(not done):
        action = agent.best_action(state, training=True)
        old_state = state
        state, _, done, _ = env.step(action)
        reward = weights.dot(self.features(old_state, action, state))
        agent.update_parameters(old_state, state, action, reward)
    return agent


  def produce_trajectory(self, agent):
    env = self.environment()
    state = env.reset() # initialize the environment
    trajectory = [state]
    done = False
    while(not done):
      action = agent.best_action(state, training=False)
      state, _, done, _ = env.step(action)
      trajectory.append(action)
      trajectory.append(state)
    return [trajectory]


######################
#### Orchestrator ####
######################

class Orchestrator():
  def __init__(self, n_arms, d, l, g, R, z):
    self.n_arms = n_arms # number of arms
    self.d = d # size of the context vector

    # lambda, array of importances over objectives; for clarity, sums to one
    # it has the same length as the number of arms
    if n_arms == 2:
      self.l = np.asarray([l, 1-l])
    else:
      self.l = l

    self.g = g # gamma, discounting rate
    self.v = R * np.sqrt(24 / z * d * np.log(1 / g)) # empirical parameter

    self.B = np.asarray([np.identity(d)]*n_arms) # n_arms covariance matrices
    self.mu = np.zeros((n_arms, d)) # n_arms means for the context features coefficients
    self.f = np.zeros((n_arms, d))


  def context(self, state):
    c1 = features_extractor.feature1(state)
    c2 = features_extractor.feature2(state)
    return np.asarray([[c1, c2]]) # size (d, 1)


  def best_arm(self, state):
    # extract n_arms vectors of d-dim vectors for context coefficients according
    # to the present beliefs
    mu_sample = np.asarray([np.random.multivariate_normal(mu[i], v * v * np.linalg.inv(B[i])) 
                            for i in range(n_arms)])
    # context vector
    c = self.context(state)

    # return the best arm
    return np.argmax(mu_sample.dot(c))


  def update_beliefs(self, old_state, rewards):
    # context in which best_arm was chosen
    c = self.context(old_state)

    # total reward; Value functions have to be added
    total_reward = rewards.dot(self.l)

    # update self.B[best_arm]
    self.B[best_arm] += B[best_arm] + np.matmul(c, c.T)

    # update self.f[best_arm]
    self.f[best_arm] += c.squeeze() * total_reward

    # update self.mu[best_arm]
    self.mu[best_arm] += np.linalg.inv(self.B[best_arm]).dot(self.f)