
import numpy as np


class RL_controller:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lr = args.lr
        self.Q_value = np.zeros((args.theta_discrete_steps, args.theta_dot_discrete_steps, 3)) # state-action values
        self.V_values = np.zeros((args.theta_discrete_steps, args.theta_dot_discrete_steps)) # state values
        self.prev_a = 0 # previous action
        # Use a previous_state = None to detect the beginning of the new round e.g. if not(self.prev_s is None): ...
        self.prev_s = None # Previous state
        

    def reset(self):
        #You need to reset sth
        self.prev_s = None
        self.prev_a = 0
        print("reset")

    def get_action(self, state, image_state, random_controller=False, episode=0):

        terminal, timestep, theta, theta_dot, reward = state
        
        if np.random.rand() > 0.8 and episode<=5000:
            action = np.random.randint(0, 3) # you have three possible actions (0,1,2)

        else:
            # print("To be implemented by student")
            # use Q values to take the best action at each state

            action_List = [self.Q_value[theta,theta_dot,0],self.Q_value[theta,theta_dot,1],self.Q_value[theta,theta_dot,2]]
            max_index = action_List.index(max(action_List))
            action = max_index

        if not(self.prev_s is None or self.prev_s == [theta, theta_dot]):
            # Calculate Q values here
            old_Q_prev = self.Q_value[self.prev_s[0],self.prev_s[1],self.prev_a]
            Q_max = max(self.Q_value[theta,theta_dot,0],self.Q_value[theta,theta_dot,1],self.Q_value[theta,theta_dot,2])
            new_Q_prev = old_Q_prev + self.lr*( reward + self.gamma*(Q_max) - old_Q_prev )
            if(new_Q_prev > self.V_values[self.prev_s[0],self.prev_s[1]]):
                self.V_values[self.prev_s[0],self.prev_s[1]] = new_Q_prev
            self.Q_value[self.prev_s[0],self.prev_s[1],self.prev_a] = new_Q_prev
            
            # pass
        #############################################################
        self.prev_s = [theta, theta_dot]
        self.prev_a = action
        # count_zeros = np.prod(self.Q_value.shape) - np.count_nonzero(self.Q_value)


        # print("Number of zeros:", count_zeros)
        # print(action)
        # print(reward)
        return [action , self.V_values]

