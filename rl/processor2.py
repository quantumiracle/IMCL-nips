import numpy as np

def process_observation(observation):  #adding relative position to pengu, to focus on movement
	"""Processes the observation as obtained from the environment for use in an agent and
	returns it.
	"""
	o = list(observation) # an array
	px = o[1]
	py = o[2]
	pvx = o[4]
	pvy = o[5]
	o = o + [o[22+i*2+1] for i in range(7)] # range: index41-47; 48dims
	for i in range(7): # head pelvis torso, toes and taluses
		o[22+i*2+0] -= px
		o[22+i*2+1] -= py
	o[18] -= px # mass pos xy made relative
	o[19] -= py
	o[20] -= pvx
	o[21] -= pvy

	o[38]/= 100
	o[39]/= 5
	o[40]/= 5
	o[1]= 0 # abs value of pel x is not relevant
	observation = o
	return observation

# expand observation from 48 to 48+14 = 62 dims   adding movement difference of time, as volicity
def transform_observation(new, old=None, step=0):
	# deal with old
	if old is None:
		old = list(process_observation(new))

	# process new
	new_processed = process_observation(new)

	# calc vel
	bodypart_velocities = [(new_processed[i]-old[i])/0.01 for i in range(22,36)]

	# substitude old with new
	for i in range(48):
		old[i] = new_processed[i]

	new_processed = new_processed + bodypart_velocities

	return new_processed, old