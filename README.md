# MORL-with-Policy-Orchestration

TO DO: 
 - [ ] In src/orchestrator.py add the value functions terms to the total_reward in self.update_beliefs(). You can approx it to the first order;
 - [ ] Test src/q_learner.py with the ram-vector;
 - [ ] Build a policy by using Q_learner changing the reward for eating a ghost to negative. It will be the expert policy;
 - [ ] Collect a set of trajectories by means of the expert policy;
 - [x] Add a way to store weights to each learning algorithm in src/;
 - [ ] Build the inverse_learner class;
 - [ ] Define in Q_Learner the features method as abstract;

### Add a way to store weights to each learning algorithm in src/
The easiest way is to save and load an object, whatever it is, by using:

```{python}
import pickle

my_obj = MyClass()

# save
file = "path/to/file"
with open(file, 'wb') as out:
  pickle.dump(object, out, pickle.HIGHEST_PROTOCOL)

# load
with open(file, 'rb') as out:
  my_obj_again = pickle.load(out)
```

