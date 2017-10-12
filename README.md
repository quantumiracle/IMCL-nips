
# IMCL-nips
Recently our team is working on the Nips2017: *learn to run* competition with Reinforcement Learning.  
Here are the techniques we use in our RL practice. 

# parallel computing with multithread and multiprocessing  
There is a balance between the actor and critic network although it is an off-policy method for DDPG.
# the reward setting 
Q value without scaling will be larger in step 100 than in step 900 in the same episode. Because <a href="https://www.codecogs.com/eqnedit.php?latex=Q_{(s)}=R&plus;\gamma&space;*Q_{(s{}')}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_{(s)}=R&plus;\gamma&space;*Q_{(s{}')}" title="Q_{(s)}=R+\gamma *Q_{(s{}')}" /></a>  
It is the Q value that trasmits the reward information from the subsequent steps to the former ones. It's also the only way for the agent to see from a globle view of the episode with such reward per step setting in this task. The more steps the agent takes, generally, the larger Q value of states in its former steps will be. So as to be consistent with our goal-a farthest running agent we want. But does this make sense? Whether or not the Q value for states with similar steps order should be scaled?  

# preprocessing of observations
The observations in this specific task can be extended to the vector representation of states in the Markov process, i.e. what variables should we use to represent the states sufficiently and efficiently. 'sufficiently' means the total number of variables is large enough to express all the information we need. 'efficiently' means there should be as little as possible redundant or overlapped information in our representation vector of states. More importantly, *there is a balance between the complexity of neural network* we used, appearing as #layers and nodes or #parameters in the network, *and the complexity of task* we're dealing with, comprising the amount of information we need to use like #dimensions of states representation vector and complexity of function mapping these inputs to outputs of the network.
