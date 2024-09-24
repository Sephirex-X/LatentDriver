This is the official repository of 

# Learning Multiple Probabilistic Decisions from Latent World Model in Autonomous Driving
Lingyu Xiao, Jiang-Jiang Liu, Sen Yang, Xiaofan Li, Xiaoqing Ye, Wankou Yang and Jingdong Wang

[[paper]()][[project page](https://sephirex-x.github.io/LatentDriver/)]
## Abstract
The autoregressive world model exhibits robust generalization capabilities in vectorized scene understanding but encounters difficulties in deriving actions due to insufficient uncertainty modeling and self-delusion. In this paper, we explore the feasibility of deriving decisions from an autoregressive world model by addressing these challenges through the formulation of multiple probabilistic hypotheses. We propose LatentDriver, a framework models the environment’s next states and the ego vehicle’s possible actions as a mixture distribution, from which a deterministic control signal is then derived. By incorporating mixture modeling, the stochastic nature of decision- making is captured. Additionally, the self-delusion problem is mitigated by providing intermediate actions sampled from a distribution to the world model. Experimental results on the recently released close-loop benchmark Waymax demonstrate that LatentDriver surpasses state-of-the-art reinforcement learning and imitation learning methods, achieving expert-level performance.

## Environment setup
coming soon