import numpy as np

class Q_Learner():
  """
  Q-learning agent for an environment where the set of possibile actions
  is finite.
  """
  def __init__(self, actions_arr, d, alpha, epsilon, g, theta=None):
    self.actions_arr = actions_arr
    self.d = d # number of features taken into account in the Q-function
    self.alpha = alpha # learning rate
    self.epsilon = epsilon # we use an epsilon-greedy strategy
    self.g = g # discounting factor

    if theta == None:
      self.theta = np.ones(d)*0.1

    elif len(theta) == d:
      self.theta = theta

    else:
      raise ValueError("theta length ({}) is different from d ({})".format(len(self.theta), d))


  def features(self, state, action):
    f1 = features_extractor.feature1(state)
    f2 = features_extractor.feature2(state)
    f3 = features_extractor.feature3(state)
    return np.asarray([f1, f2, f3])


  def Q(self, state, action):
    return self.features(state, action).dot(self.theta)


  def best_action(self, state, training=False):

    u = np.random.uniform()

    if u > self.epsilon and training:
      return self.actions_arr[np.random.randint(len(self.actions_arr))]

    else:
      table = np.asarray([self.Q(new_state, action) for action in self.actions_arr])
      return self.actions_arr[np.argmax(table)]


  def difference(self, old_state, new_state, action, reward):
    return reward + self.g * self.best_action(new_state) - self.Q(old_state, action)


  def update_parameters(self, old_state, new_state, action, reward):
    diff = self.difference(old_state, new_state, action, reward)
    self.theta += self.alpha * diff * self.features(old_state, action)