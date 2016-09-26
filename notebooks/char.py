data_fname = '../wnut_ner_evaluation/data/train_notypes'
xs, ys = [], []
with open(data_fname, 'r') as f:
    x, y = [], []
    for i, line in enumerate(f):
        split = line.split()
        if split:
            x.append(split[0])
            y.append(split[1])
        else: 
            xs.append(x)
            ys.append(y)
            x, y = [], []

data_fname = '../wnut_ner_evaluation/data/dev_notypes'
dev_xs, dev_ys = [], []
with open(data_fname, 'r') as f:
    x, y = [], []
    for i, line in enumerate(f):
        split = line.split()
        if split:
            x.append(split[0])
            y.append(split[1])
        else: 
            dev_xs.append(x)
            dev_ys.append(y)
            x, y = [], []

class EntitySegmenter():
    def __init__(self, session, **hyperparams):
        print "Model Loading...",
        self.hyperparams = hyperparams
        for var, val in hyperparams.items():
            setattr(self, var, val)
        
        self.manipulate_params()
        self.build_forward()
        self.build_loss()
        self.build_optimizer()
        
        self.session = session
        self.session.run(tf.initialize_all_variables())
        print "Done"
        
    def manipulate_params(self):
        pass
    
    def build_forward(self):
        # inputs
        self.x_input = tf.placeholder(tf.int32, [None, self.max_time, self.max_chargrams])
        self.x_weight = tf.placeholder(tf.float32, [None, self.max_time, self.max_chargrams])
        self.y_input = tf.placeholder(tf.int32, [None, self.max_time])
        self.dropout_keep = tf.placeholder(tf.float32)
        
        # embed and take weighted sum of character grams as word embedding
        self.chargram_vectors = tf.Variable(tf.random_uniform([self.xvocab.n, self.char_embed_size], 
                                                             -.1, .1, tf.float32))
        self.embedded_chargrams = tf.nn.embedding_lookup(self.chargram_vectors, 
                                                         self.x_input)
        tile_weights = tf.tile(tf.expand_dims(self.x_weight, [3]), [1,1,1,self.char_embed_size])
        self.embedded_words = tf.reduce_sum(tile_weights * self.embedded_chargrams, [2])
        
    
        # mlp
        self.W = tf.Variable(tf.random_uniform([self.char_embed_size, self.yvocab.n], 
                                              -.1, .1, tf.float32))
        self.b = tf.Variable(tf.zeros([self.yvocab.n]))
        
        logits = tf.matmul(tf.reshape(self.embedded_words, [-1, self.char_embed_size]), self.W) + self.b
        self.logits = tf.reshape(logits, [-1, self.max_time, self.yvocab.n])
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
        
    def batch_generator(self, x, y, batch_size=64):
        # shuffle the data
        xy = zip(x,y)
        random.shuffle(xy)
        x, y = zip(*xy)
        
        # create minibatches
        num_batches = len(y) // batch_size + 1
        for batch_i in range(num_batches):
            start, end = batch_i*batch_size, (batch_i+1)*batch_size
            yield batch_i, x[start:end], y[start:end]
    
    def print_progress(epoch_i, n_epoch, batch_i, loss):
        if batch_i == 0:
            print
        print "\r Epoch {0}/{1} : Batch {2} : Loss {3:.4f}".format(
              epoch_i+1, n_epoch, batch_i+1, loss),
    def fit(self, x, y, n_epoch=1, batch_size=64, **partial_fit_kwargs):
        for epoch_i in range(n_epoch):
            for batch_i, batch_x, batch_y in self.batch_generator(x, y, batch_size): 
                loss = self.partial_fit(batch_x, batch_y)
                self.print_progress(loss, batch_i, epoch_i)
                
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
    
    def save(self):
        pass
    
    def load(self):
        pass