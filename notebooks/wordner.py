from tfmodel import *

class WordNER(TFModel):
    def build_forward(self):
        # inputs
        self.x_input = tf.placeholder(tf.int32, [None, self.max_time])
        self.x_weight = tf.placeholder(tf.float32, [None, self.max_time])
        self.y_input = tf.placeholder(tf.int32, [None, self.max_time])
        self.dropout_keep = tf.placeholder(tf.float32)
        
        # embed and take weighted sum of character grams as word embedding
        self.word_vectors = tf.Variable(tf.random_uniform([self.xvocab.v, self.word_embed_size], 
                                                             -.1, .1, tf.float32))
        self.embedded_words = tf.nn.embedding_lookup(self.word_vectors, 
                                                         self.x_input)
         
        # hidden layer
        self.Wh = tf.Variable(tf.random_uniform([self.word_embed_size, self.hidden_size], 
                                              -.1, .1, tf.float32))
        self.bh = tf.Variable(tf.zeros([self.hidden_size]))
        hidden = tf.matmul(tf.reshape(self.embedded_words, [-1, self.word_embed_size]), self.Wh) + self.bh
        hidden = tf.nn.relu(hidden)
        
        # softmax layer
        self.W = tf.Variable(tf.random_uniform([self.hidden_size, self.yvocab.v], 
                                              -.1, .1, tf.float32))
        self.b = tf.Variable(tf.zeros([self.yvocab.v]))
        
        logits = tf.matmul(hidden, self.W) + self.b
        self.logits = tf.reshape(logits, [-1, self.max_time, self.yvocab.v])
        self.labels = tf.argmax(self.logits, 2)
        
    def build_loss(self):
        # convert logits to lists
        logits_list = [tf.squeeze(t, [1]) for t in tf.split(1, self.max_time, self.logits)]
        targets_list = [tf.squeeze(t, [1]) for t in tf.split(1, self.max_time, self.y_input)]
        weights_list = [tf.squeeze(t, [1]) for t in tf.split(1, self.max_time, self.x_weight)]
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