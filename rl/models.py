import tensorflow as tf
import tensorlayer as tl

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Actor(Model):
    def __init__(self, act_dims, name):
        super(Actor, self).__init__(name=name)
        self.act_dims = act_dims

        self.action_multiplier = 0.5
        self.action_bias = 0.5

    def __call__(self, actor_input, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
                tl.layers.set_name_reuse(reuse)

            actor = tl.layers.InputLayer(actor_input, name='actor_input_layer')
            actor = tl.layers.DenseLayer(actor, n_units=96, name='dense_layer1',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.ReshapeLayer(actor, [-1, 96, 1], name ='reshape_layer')     
            actor = tl.layers.Conv1dLayer(actor, shape=[5, 1, 8], name='conv_layer1', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.Conv1dLayer(actor, shape=[3, 8, 4], name='conv_layer2', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.Conv1dLayer(actor, shape=[3, 4, 2], name='conv_layer3', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.FlattenLayer(actor,name ='flatten_layer')
            actor = tl.layers.DenseLayer(actor, n_units=128, name='dense_layer2',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.DenseLayer(actor, n_units=48, name='dense_layer3',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.DenseLayer(actor, n_units=self.act_dims, act=tf.nn.tanh, name='dense_layer4')
            actor = tl.layers.LambdaLayer(actor, lambda a: a*self.action_multiplier + self.action_bias, name='lambda_layer')
            actor_output = actor.outputs
            print 'action shape', actor_output.get_shape()
            # actor.print_layers()
            # actor.print_params()
        return actor_output

class Critic(Model):
    def __init__(self, name):
        super(Critic, self).__init__(name=name)

    def __call__(self, critic_obs_input, critic_act_input, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
                tl.layers.set_name_reuse(reuse)
            
            critic1 = tl.layers.InputLayer(critic_obs_input, name='obs_input_layer')
            critic2 = tl.layers.InputLayer(critic_act_input, name='act_input_layer')
            critic = tl.layers.ConcatLayer(layer=[critic1, critic2], concat_dim=1, name='concat_input_layer')
            critic = tl.layers.DenseLayer(critic, n_units=128, name='dense_layer1',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.ReshapeLayer(critic, [-1, 128, 1], name ='reshape_layer')
            critic = tl.layers.Conv1dLayer(critic, shape=[5, 1, 8], name='conv_layer1', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.Conv1dLayer(critic, shape=[3, 8, 4], name='conv_layer2', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.Conv1dLayer(critic, shape=[3, 4, 2], name='conv_layer3', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.FlattenLayer(critic,name ='flatten_layer')
            critic = tl.layers.DenseLayer(critic, n_units=128, name='dense_layer2',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.DenseLayer(critic, n_units=64, name='dense_layer3',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.DenseLayer(critic, n_units=1, name='dense_layer4' )
            critic_output = critic.outputs
            print 'q-value shape', critic_output.get_shape()
            # critic.print_layers()
            # critic.print_params()
        return critic_output
