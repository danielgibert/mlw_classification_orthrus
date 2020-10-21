import tensorflow as tf

class BimodalCNN(tf.keras.Model):
    """
    The method presented in "Orthrus" differs in the architecture used for the opcodes component.
    This is a simplified architecture (same os the opcodes part). It achieves better results though.
    Feel free to implement the original one if needed.
    """
    def __init__(self, parameters):
        super(BimodalCNN, self).__init__()
        self.parameters = parameters

    def build(self, input_shapes):
        # Bytes component
        ######################################### Bytes component ######################################################
        self.bytes_emb = tf.keras.layers.Embedding(self.parameters['bytes']['V'],
                                                   self.parameters['bytes']['E'],
                                                   input_shape=(None,
                                                                self.parameters['bytes']['seq_length']))

        self.bytes_conv_3 = tf.keras.layers.Conv2D(self.parameters['bytes']['conv']['num_filters'],
                                             (self.parameters['bytes']['conv']['size'][0], self.parameters['bytes']['E']),
                                             activation="relu",
                                             input_shape=(None,
                                                          self.parameters['bytes']['seq_length'],
                                                          self.parameters['bytes']['E']))
        self.bytes_global_max_pooling_3 = tf.keras.layers.GlobalMaxPooling2D()

        self.bytes_conv_5 = tf.keras.layers.Conv2D(self.parameters['bytes']['conv']['num_filters'],
                                             (self.parameters['bytes']['conv']['size'][1], self.parameters['E']),
                                             activation="relu",
                                             input_shape=(None,
                                                          self.parameters['bytes']['seq_length'],
                                                          self.parameters['bytes']['E']))
        self.bytes_global_max_pooling_5 = tf.keras.layers.GlobalMaxPooling2D()


        self.bytes_conv_7 = tf.keras.layers.Conv2D(self.parameters['bytes']['conv']['num_filters'],
                                             (self.parameters['bytes']['conv']['size'][2], self.parameters['E']),
                                             activation="relu",
                                             input_shape=(None,
                                                          self.parameters['bytes']['seq_length'],
                                                          self.parameters['bytes']['E']))
        self.bytes_global_max_pooling_7 = tf.keras.layers.GlobalMaxPooling2D()

        ####################################### Opcodes component ######################################################
        self.opcodes_emb = tf.keras.layers.Embedding(self.parameters['opcodes']['V'],
                                                   self.parameters['opcodes']['E'],
                                                   input_shape=(None,
                                                                self.parameters['opcodes']['seq_length']))

        self.opcodes_conv_3 = tf.keras.layers.Conv2D(self.parameters['opcodes']['conv']['num_filters'],
                                                   (self.parameters['opcodes']['conv']['size'][0],
                                                    self.parameters['opcodes']['E']),
                                                   activation="relu",
                                                   input_shape=(None,
                                                                self.parameters['opcodes']['seq_length'],
                                                                self.parameters['opcodes']['E']))
        self.opcodes_global_max_pooling_3 = tf.keras.layers.GlobalMaxPooling2D()

        self.opcodes_conv_5 = tf.keras.layers.Conv2D(self.parameters['opcodes']['conv']['num_filters'],
                                                   (self.parameters['opcodes']['conv']['size'][1], self.parameters['E']),
                                                   activation="relu",
                                                   input_shape=(None,
                                                                self.parameters['opcodes']['seq_length'],
                                                                self.parameters['opcodes']['E']))
        self.opcodes_global_max_pooling_5 = tf.keras.layers.GlobalMaxPooling2D()

        self.opcodes_conv_7 = tf.keras.layers.Conv2D(self.parameters['opcodes']['conv']['num_filters'],
                                                   (self.parameters['opcodes']['conv']['size'][2], self.parameters['E']),
                                                   activation="relu",
                                                   input_shape=(None,
                                                                self.parameters['opcodes']['seq_length'],
                                                                self.parameters['opcodes']['E']))
        self.opcodes_global_max_pooling_7 = tf.keras.layers.GlobalMaxPooling2D()


        self.dense_dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(self.parameters['output'],
                                           activation="softmax")

    def call(self, opcodes_tensor, bytes_tensor, training=False):
        bytes_emb = self.bytes_emb(bytes_tensor)
        bytes_emb_expanded = tf.keras.backend.expand_dims(bytes_emb, axis=-1)

        bytes_conv_3 = self.bytes_conv_3(bytes_emb_expanded)
        bytes_pool_3 = self.bytes_global_max_pooling_3(bytes_conv_3)

        bytes_conv_5 = self.bytes_conv_5(bytes_emb_expanded)
        bytes_pool_5 = self.bytes_global_max_pooling_5(bytes_conv_5)

        bytes_conv_7 = self.bytes_conv_7(bytes_emb_expanded)
        bytes_pool_7 = self.bytes_global_max_pooling_7(bytes_conv_7)

        opcodes_emb = self.opcodes_emb(opcodes_tensor)
        opcodes_emb_expanded = tf.keras.backend.expand_dims(opcodes_emb, axis=-1)

        opcodes_conv_3 = self.opcodes_conv_3(opcodes_emb_expanded)
        opcodes_pool_3 = self.opcodes_global_max_pooling_3(opcodes_conv_3)

        opcodes_conv_5 = self.opcodes_conv_5(opcodes_emb_expanded)
        opcodes_pool_5 = self.opcodes_global_max_pooling_5(opcodes_conv_5)

        opcodes_conv_7 = self.opcodes_conv_7(opcodes_emb_expanded)
        opcodes_pool_7 = self.opcodes_global_max_pooling_7(opcodes_conv_7)

        features = tf.keras.layers.concatenate([bytes_pool_3, bytes_pool_5, bytes_pool_7, opcodes_pool_3, opcodes_pool_5, opcodes_pool_7])
        features_dropout = self.dense_dropout(features, training=training)
        output = self.dense(features_dropout)

        return output

    def load_opcodes_subnetwork_pretrained_weights(self, model):
        """
        Loads the pretrained weights of the opcodes subnetwork into the bimodal architecture
        :param model: filepath to the opcodes' model
        :return:
        """
        print("ToImplement")

    def load_bytes_subnetwork_pretrained_weights(self, model):
        """
        Loads the pretrained weights of the bytes subnetwork into the bimodal architecture
        :param model: filepath to the opcodes' model
        :return:
        """
        print("ToImplement")