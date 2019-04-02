from orchestrator import Orchestrator

def main():
  orc = Orchestrator(n_arms=2, d=3, l=0.7, g=0.9, R=1, z=1)
  print(orc.context(1))

if __name__ == '__main__':
  main()