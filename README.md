
# IMCL-nips
Recently our team is working on the Nips2017: *learn to run* competition with Reinforcement Learning.  
Here are the techniques we use in our RL practice. 

# parallel computing with multithread and multiprocessing  
There is a balance between the actor and critic network although it is an off-policy method for DDPG.
# the reward setting 
Q value without scaling will be larger in step 100 than in step 900 in the same episode. Because <a href="https://www.codecogs.com/eqnedit.php?latex=Q_{(s)}=R&plus;\gamma&space;*Q_{(s{}')}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_{(s)}=R&plus;\gamma&space;*Q_{(s{}')}" title="Q_{(s)}=R+\gamma *Q_{(s{}')}" /></a>  
It is the Q value that trasmits the reward information from the subsequent steps to the former ones. It's also the only way for the agent to see from a globle view of the episode with such reward per step setting in this task. The more steps the agent takes, generally, the larger Q value of states in its former steps will be. So as to be consistent with our goal-a farthest running agent we want. But does this make sense? Whether or not the Q value for states with similar steps order should be scaled?
