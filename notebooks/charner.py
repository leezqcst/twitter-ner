from tfmodel import *

class ChargramNER(TFModel):
    def build_forward(self):
        # inputs
        self.x_input = tf.placeholder(tf.int32, [None, self.max_time, self.max_chargrams])
        self.x_weight = tf.placeholder(tf.float32, [None, self.max_time, self.max_chargrams])
        self.y_input = tf.placeholder(tf.int32, [None, self.max_time])
        self.dropout_keep = tf.placeholder(tf.float32)
        
        # embed and take weighted sum of character grams as word embedding
        self.chargram_vectors = tf.Variable(tf.random_uniform([self.xvocab.v, self.char_embed_size], 
                                                             -.1, .1, tf.float32))
        self.embedded_chargrams = tf.nn.embedding_lookup(self.chargram_vectors, 
                                                         self.x_input)
        tile_weights = tf.tile(tf.expand_dims(self.x_weight, [3]), [1,1,1,self.char_embed_size])
        self.embedded_words = tf.reduce_sum(tile_weights * self.embedded_chargrams, [2])
        
    
        # hidden layer
        self.Wh = tf.Variable(tf.random_uniform([self.char_embed_size, self.hidden_size], 
                                              -.1, .1, tf.float32))
        self.bh = tf.Variable(tf.zeros([self.hidden_size]))
        hidden = tf.matmul(tf.reshape(self.embedded_words, [-1, self.char_embed_size]), self.Wh) + self.bh
        hidden = tf.nn.relu(hidden)
        
        # softmax layer
        self.W = tf.Variable(tf.random_uniform([self.hidden_size, self.yvocab.v], 
                                              -.1, .1, tf.float32))
        self.b = tf.Variable(tf.zeros([self.yvocab.v]))
        
        logits = tf.matmul(hidden, self.W) + self.b
        self.logits = tf.reshape(logits, [-1, self.max_time, self.yvocab.v])
        self.labels = tf.argmax(self.logits, 2)
        
    def build_loss(self):
        word_weights = tf.minimum(tf.reduce_sum(self.x_weight, [2]), 1.)
        # convert logits to lists
        logits_list = [tf.squeeze(t, [1]) for t in tf.split(1, self.max_time, self.logits)]
        targets_list = [tf.squeeze(t, [1]) for t in tf.split(1, self.max_time, self.y_input)]
        weights_list = [tf.squeeze(t, [1]) for t in tf.split(1, self.max_time, word_weights)]
        seq_loss = tf.nn.seq2seq.sequence_loss_by_example(logits_list,
                                                          targets_list,
                                                          weights_list)
        self.loss = tf.reduce_mean(seq_loss)
    
    def build_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def partial_fit(self, x, y, measure_only=False):
        x_input, x_weight = zip(*x)
        feed_dict = {
            self.x_input:x_input,
            self.x_weight:x_weight,
            self.y_input:y,
            self.dropout_keep:1.-self.dropout
        }
        
        if measure_only:
            loss = self.session.run(self.loss, feed_dict)
        else:
            loss, _ = self.session.run([self.loss, self.train_op], feed_dict)
        return loss
    
    def predict(self, x):
        x_input, x_weight = zip(*x)
        feed_dict = {
            self.x_input:x_input,
            self.x_weight:x_weight,
            self.dropout_keep:1.
        }
        predictions = self.session.run(self.labels, feed_dict)
        return predictions