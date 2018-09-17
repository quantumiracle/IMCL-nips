import multiprocessing as mp
import threading as th
import random
import time
import traceback

from multiprocessing import Process, Pipe, Queue
from osim.env import RunEnv


ncpu = 4  #not used ;used for interacting with env to sample 
# ncpu = multiprocessing.cpu_count()

plock = mp.Lock()
tlock = th.Lock()

env_id = int(random.random()*100000)

def get_env_id():
    global env_id, tlock
    tlock.acquire()
    id = env_id 
    env_id += 1
    tlock.release()
    return id 

def separate_env(pq, cq, plock):
    plock.acquire()
    print('Starting a Separate Env',pq ,cq)
    
    try:
        env = RunEnv(visualize=False)    #only run the env, but without agent interacting with env 
    except Exception as e:
        print('Error on starting the Env')
        traceback.print_exc()
        plock.release()
        return
    else:
        plock.release()

    def floatify(np):
        return [float(np[i]) for i in range(len(np))]
    
    try:
        while True:
            msg = pq.get()
            if msg[0] == 'reset':
                observation = env.reset(difficulty=2)
                cq.put(floatify(observation))
            elif msg[0] == 'step':
                observation, reward, done, info = env.step(msg[1])     #interacting with env
                observation = floatify(observation)
                cq.put((observation, reward, done, info))
            else:
                cq.close()
                pq.close()
                del env
                break
    except Exception as e:
        traceback.print_exc()
        print('(SeparateEnv) got error!!!')
        cq.put(('error',e))

    return        

class EnvIns():  #use env_process to create a new process and reset to stop it; one EnvIns.id with one env_id number
    def __init__(self):
        self.occupied = False
        self.id = get_env_id()
        self.printinfo('EnvIns is created!')
        self.lock = th.Lock()
        self.pq, self.cq = Queue(1), Queue(1)

        self.newprocess()

    def printinfo(self,s):
        print(('(EnvIns) {} ').format(self.id)+str(s))

    def timer_update(self):
        self.last_interaction = time.time()

    # create a new RunEnv in a new process.
    def newprocess(self):
        global plock
        self.timer_update()

        self.env_process = Process(target=separate_env, 
                                args=(self.pq, self.cq, plock))
        self.env_process.daemon = True
        self.env_process.start()

        self.reset_count = 0
        self.step_count = 0
        self.timer_update()
        return

    def is_occupied(self):
        if self.occupied == False:
            return False
        else:
            if time.time() - self.last_interaction > 20*60:
                self.printinfo('No interaction for a long time, release now. Apply for a new id.')
                self.id = get_env_id()
                self.occupied = False
                return False
            else:
                return True
    
    def occupy(self):
        self.lock.acquire()
        if self.is_occupied() == False:
            self.occupied = True
            self.id = get_env_id()
            self.lock.release()
            return True
        else:
            self.lock.release()
            return False

    def release(self):
        self.lock.acquire()
        self.occupied = False
        self.id = get_env_id()
        self.lock.release()
    
    # send x to the process.
    def send(self, x):
        self.pq.put(x)

    # receive from the process.
    def receive(self):
        rec_info = self.cq.get()

        if rec_info[0] == 'error':
            einfo = rec_info[1]
            self.printinfo('Got Exception')
            self.printinfo(einfo)
            raise Exception(einfo)
        
        return rec_info

    def kill(self):  #as waiting(other processes) until it finished 
        if not self.env_process.is_alive():
            self.printinfo('Process already dead!')
        else:
            self.send(('exit',))
            self.printinfo('Waiting for join() ...')
            while 1:
                self.env_process.join(timeout=5)  #block current process until the one induces join finished (i.e. env_process in this program)
                if not self.env_process.is_alive():
                    break
                else:
                    self.printinfo('Process is not joining after 5s, still waiting...')
            self.printinfo('Process has joined!')
    
    def reset(self):
        self.timer_update()

        if not self.env_process.is_alive():
            self.printinfo('Process found dead on reset(). reloading.')
            self.kill()
            self.newprocess()

        if self.reset_count>50 or self.step_count>10000:
            self.printinfo('Environment has been resetted too much. Memory leaks and other problems might present. reloading.')
            self.kill()
            self.newprocess()

        self.reset_count += 1
        self.send(('reset',))
        r = self.receive()
        self.timer_update()

        return r

    def step(self, actions):
        self.timer_update()
        self.send(('step',actions))
        r = self.receive()
        self.timer_update()
        self.step_count += 1
        return r

    def __del__(self):
        self.printinfo('__del__')
        self.kill()
        self.printinfo('__del__ accomplished.')

    
class EnvPool():  #each content of EnvPool is a EnvIns--one process class
    def __init__(self, n=1):
        self.printinfo('Starting '+str(n)+' instance(s)...')
        self.pool = [EnvIns() for i in range(n)]
        self.lock = th.Lock()

    def printinfo(self, s):
        print(('(EnvPool)')+str(s))

    def acquire_env(self):   #choose a process
        self.lock.acquire()
        for e in self.pool:
            if e.occupy() == True:
                self.lock.release()
                return e    
        self.lock.release()
        return False

    def release_env(self, ei):
        self.lock.acquire()
        for e in self.pool:
            if e == ei:
                e.release()
        self.lock.release()

    def get_env_by_id(self, id):
        for e in self.pool:
            if e.id == id:
                return e
        return False

    def __del__(self):
        for e in self.pool:
            del e
    
    