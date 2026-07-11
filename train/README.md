1. interact with PortfolioEnv for N steps
2. store trajectories
3. compute advantages
4. update policy using clipped objective
5. repeat

So sth like: repeat until N environment steps:
    Collect rollout
          ↓
    Compute advantages
          ↓
    Update policy
          ↓
    Every 10,000 steps:
        Evaluate on validation set
        Save best model

use ´tensorboard --logdir logs/tensorboard` for logs
