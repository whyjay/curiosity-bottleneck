import numpy as np
import tensorflow as tf
from baselines import logger
from utils import fc, conv, ortho_init, deconv
from stochastic_policy import StochasticPolicy
from tf_util import get_available_gpus
from mpi_util import RunningMeanStd


def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]:
        size *= shapel.value
    return tf.reshape(x, (-1, size))


def undo2d(x, shape):
    size = 1
    for shapel in shape:
        size *= shapel
    assert size == x.get_shape()[1]
    return tf.reshape(x, (-1, *shape))


def _fcnobias(x, scope, nh, *, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        return tf.matmul(x, w)


def _normalize(x):
    eps = 1e-5
    mean, var = tf.nn.moments(x, axes=(-1,), keepdims=True)
    return (x - mean) / tf.sqrt(var + eps)


class CnnPolicy(StochasticPolicy):
    def __init__(self, scope, ob_space, ac_space,
                 policy_size='normal', maxpool=False, extrahid=True, hidsize=128, memsize=128, rec_gate_init=0.0,
                 update_ob_stats_independently_per_gpu=True,
                 proportion_of_exp_used_for_predictor_update=1.,
                 exploration_type='bottleneck', beta=0.001, rew_counter=None
                 ):
        StochasticPolicy.__init__(self, scope, ob_space, ac_space)

        self.proportion_of_exp_used_for_predictor_update = proportion_of_exp_used_for_predictor_update
        enlargement = {
            'small': 1,
            'normal': 2,
            'large': 4
        }[policy_size]
        rep_size = 512
        self.ph_mean = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obmean")
        self.ph_std = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obstd")
        memsize *= enlargement
        hidsize *= enlargement
        convfeat = 16*enlargement
        self.ob_rms = RunningMeanStd(shape=list(ob_space.shape[:2])+[1], use_mpi=not update_ob_stats_independently_per_gpu)
        ph_istate = tf.placeholder(dtype=tf.float32,shape=(None,memsize), name='state')
        pdparamsize = self.pdtype.param_shape()[0]
        self.memsize = memsize

        #Inputs to policy and value function will have different shapes depending on whether it is rollout
        #or optimization time, so we treat separately.
        self.pdparam_opt, self.vpred_int_opt, self.vpred_ext_opt, self.snext_opt = \
            self.apply_policy(self.ph_ob[None][:,:-1],
                              reuse=False,
                              scope=scope,
                              hidsize=hidsize,
                              memsize=memsize,
                              extrahid=extrahid,
                              sy_nenvs=self.sy_nenvs,
                              sy_nsteps=self.sy_nsteps - 1,
                              pdparamsize=pdparamsize
                              )
        self.pdparam_rollout, self.vpred_int_rollout, self.vpred_ext_rollout, self.snext_rollout = \
            self.apply_policy(self.ph_ob[None],
                              reuse=True,
                              scope=scope,
                              hidsize=hidsize,
                              memsize=memsize,
                              extrahid=extrahid,
                              sy_nenvs=self.sy_nenvs,
                              sy_nsteps=self.sy_nsteps,
                              pdparamsize=pdparamsize
                              )

        self.exploration_type = exploration_type
        self.max_table = 0

        self.define_bottleneck_rew(convfeat=convfeat, rep_size=rep_size/8, enlargement=enlargement, beta=beta, rew_counter=rew_counter)

        pd = self.pdtype.pdfromflat(self.pdparam_rollout)

        self.a_samp = pd.sample()
        self.nlp_samp = pd.neglogp(self.a_samp)
        self.entropy_rollout = pd.entropy()
        self.pd_rollout = pd

        self.pd_opt = self.pdtype.pdfromflat(self.pdparam_opt)
        self.a_samp_opt = self.pd_opt.sample()

        self.ph_istate = ph_istate

        self.scope = scope

        # for gradcam policy
        a_one_hot = tf.one_hot(self.ph_ac, self.ac_space.n, axis=2)
        loss_cam_pol = tf.reduce_mean(tf.multiply(self.pdparam_opt, a_one_hot))
        self.conv_out = tf.get_default_graph().get_tensor_by_name('ppo/pol/Relu_2:0')
        self.grads = tf.gradients(loss_cam_pol, self.conv_out)[0]
        # for gradcam aux
        loss_cam_aux = self.kl
        if int(str(tf.__version__).split('.')[1]) < 10:
            self.conv_aux_out = tf.get_default_graph().get_tensor_by_name('ppo/LeakyRelu_2/Maximum:0')
        else:
            self.conv_aux_out = tf.get_default_graph().get_tensor_by_name('ppo/LeakyRelu_2:0')
        self.grads_aux = tf.abs(tf.gradients(loss_cam_aux, self.conv_aux_out)[0])

        weights = tf.reduce_mean(tf.reduce_mean(self.grads, 2), 1)
        weights = tf.expand_dims(tf.expand_dims(weights, axis=1), axis=1)
        weights = tf.tile(weights, [1, 6, 6, 1])
        cams = tf.reduce_sum((weights * self.conv_out), axis=3)
        self.cams = tf.maximum(cams, tf.zeros_like(cams))

        weights_aux = tf.reduce_mean(tf.reduce_mean(self.grads_aux, 2), 1)
        weights_aux = tf.expand_dims(tf.expand_dims(weights_aux, axis=1), axis=1)
        weights_aux = tf.tile(weights_aux, [1, 7, 7, 1])
        cams_aux = tf.nn.relu(tf.reduce_sum((weights_aux * self.conv_aux_out), axis=3))
        self.cams_aux = tf.maximum(cams_aux, tf.zeros_like(cams_aux))

    @staticmethod
    def apply_policy(ph_ob, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize):
        data_format = 'NHWC'
        ph = ph_ob
        assert len(ph.shape.as_list()) == 5  # B,T,H,W,C
        logger.info("CnnPolicy: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        X = tf.cast(ph, tf.float32) / 255.
        X = tf.reshape(X, (-1, *ph.shape.as_list()[-3:]))

        activ = tf.nn.relu
        yes_gpu = any(get_available_gpus())
        with tf.variable_scope(scope, reuse=reuse), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):
            X = activ(conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), data_format=data_format))
            X = to2d(X)
            mix_other_observations = [X]
            X = tf.concat(mix_other_observations, axis=1)
            X = activ(fc(X, 'fc1', nh=hidsize, init_scale=np.sqrt(2)))
            additional_size = 448
            X = activ(fc(X, 'fc_additional', nh=additional_size, init_scale=np.sqrt(2)))
            snext = tf.zeros((sy_nenvs, memsize))
            mix_timeout = [X]

            Xtout = tf.concat(mix_timeout, axis=1)
            if extrahid:
                Xtout = X + activ(fc(Xtout, 'fc2val', nh=additional_size, init_scale=0.1))
                X     = X + activ(fc(X, 'fc2act', nh=additional_size, init_scale=0.1))
            pdparam = fc(X, 'pd', nh=pdparamsize, init_scale=0.01)
            vpred_int   = fc(Xtout, 'vf_int', nh=1, init_scale=0.01)
            vpred_ext   = fc(Xtout, 'vf_ext', nh=1, init_scale=0.01)

            pdparam = tf.reshape(pdparam, (sy_nenvs, sy_nsteps, pdparamsize))
            vpred_int = tf.reshape(vpred_int, (sy_nenvs, sy_nsteps))
            vpred_ext = tf.reshape(vpred_ext, (sy_nenvs, sy_nsteps))
        return pdparam, vpred_int, vpred_ext, snext

    def define_bottleneck_rew(self, convfeat, rep_size, enlargement, beta=1e-2, rew_counter=None):
        logger.info("Using Curiosity Bottleneck ****************************************************")
        v_target = tf.reshape(self.ph_ret_ext, (-1, 1))

        if rew_counter is None:
            sched_coef = 1.
        else:
            sched_coef = tf.minimum(rew_counter/1000, 1.)

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]
                mu = fc(rgbr[0], 'fc_mu', nh=rep_size, init_scale=np.sqrt(2))
                sigma = tf.nn.softplus(fc(rgbr[0], 'fc_sigma', nh=rep_size, init_scale=np.sqrt(2)))
                z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
                v = fc(z, 'value', nh=1, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(sigma)
        self.max_feat = tf.reduce_max(tf.abs(z))

        self.kl = 0.5 * tf.reduce_sum(
            tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1,
            axis=-1, keep_dims=True)
        self.int_rew = tf.stop_gradient(self.kl)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        self.aux_loss_raw = sched_coef * tf.square(v_target - v) + beta * self.kl
        # self.aux_loss_raw = beta * self.kl
        self.aux_loss = sched_coef * tf.square(v_target - v) + beta * self.kl
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)
        self.v_int = v

    def initial_state(self, n):
        return np.zeros((n, self.memsize), np.float32)

    def call(self, dict_obs, new, istate, update_obs_stats=False):
        """
        called when step()
        """
        for ob in dict_obs.values():
            if ob is not None:
                if update_obs_stats:
                    raise NotImplementedError
                    ob = ob.astype(np.float32)
                    ob = ob.reshape(-1, *self.ob_space.shape)
                    self.ob_rms.update(ob)
        # Note: if it fails here with ph vs observations inconsistency, check if you're loading agent from disk.
        # It will use whatever observation spaces saved to disk along with other ctor params.
        feed1 = { self.ph_ob[k]: dict_obs[k][:,None] for k in self.ph_ob_keys }
        feed2 = { self.ph_istate: istate, self.ph_new: new[:,None].astype(np.float32) }
        feed1.update({self.ph_mean: self.ob_rms.mean, self.ph_std: self.ob_rms.var ** 0.5})
        a, vpred_int,vpred_ext, nlp, newstate, ent = tf.get_default_session().run(
            [self.a_samp, self.vpred_int_rollout, self.vpred_ext_rollout, self.nlp_samp, self.snext_rollout, self.entropy_rollout],
            feed_dict={**feed1, **feed2})
        return a[:,0], vpred_int[:,0],vpred_ext[:,0], nlp[:,0], newstate, ent[:,0]

