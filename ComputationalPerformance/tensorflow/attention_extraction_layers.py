from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.initializers import Constant


class AttentionWeights(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        # self.init = initializers.get(Constant(value=1))

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionWeights, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        return a

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

    def get_config(self):
        config={'step_dim':self.step_dim}
        base_config = super(AttentionWeights, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ContextVector(Layer):
    def __init__(self, **kwargs):
        super(ContextVector, self).__init__(**kwargs)
        self.features_dim = 0

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.features_dim = input_shape[0][-1]
        self.built = True

    def call(self, x, **kwargs):
        assert len(x) == 2
        h = x[0]
        a = x[1]
        a = K.expand_dims(a)
        weighted_input = h * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.features_dim

    def get_config(self):
        base_config = super(ContextVector, self).get_config()
        return dict(list(base_config.items()))
