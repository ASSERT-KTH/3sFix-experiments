import tensorflow as tf
import numpy as np
import math
import collections
import random
import pickle
import glob,os
from tempfile import gettempdir
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

current_data = []
current_data_index = 0
dataCounter = 0
allData = []

def generate_batch(batch_size, num_skips, skip_window):
    global current_data_index
    global current_data
    global dataCounter
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    while(len(current_data) < 3):
        current_data = allData[dataCounter]
        dataCounter = (dataCounter + 1) % len(allData)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    if(len(current_data) < span): # In case of current data is too short
        skip_window = (len(current_data) // 2) - (1 - len(current_data) % 2)
        num_skips = skip_window*2
        span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span) # Buffer for current window
    if current_data_index + span > len(current_data): # Reset index if exceeding length
        current_data_index = 0
    buffer.extend(current_data[current_data_index:current_data_index + span]) # Load current window
    current_data_index += span # Update index
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window] # Words around the selected word
        words_to_use = random.sample(context_words, num_skips) # Random sample from these
        for j, context_word in enumerate(words_to_use): # Create batch and coresspoding label
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if current_data_index >= len(current_data):
            current_data = allData[dataCounter]
            dataCounter = (dataCounter + 1) % len(allData)
            # See https://github.com/tensorflow/tensorflow/issues/10866
            # Original buffer[:] = data[:span], cause TypeError
            for word in current_data[:span]:
                buffer.append(word)
            current_data_index = span
        else:
            buffer.append(current_data[current_data_index])
            current_data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    current_data_index = (current_data_index + len(current_data) - span) % len(current_data)
    return batch, labels

# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

def main():
    global current_data
    global allData
    global dataCounter

    with open("PATH_TO_STORE_THE_DICTIONARY" , "rb") as f:
        [count,dictionary,reverse_dictionary,vocabulary_size] = pickle.load(f)
    print("Loaded count, dictionary, reverse_dictionary and vocabulary_size")

    index_path = "PATH_TO_STORE_INDEXED_PROEJCTS"
    for filename in glob.glob(os.path.join(index_path, "*.pickle")):
        with open(filename, "rb") as file:
            allData.append(pickle.load(file))
    print("Loaded all data")

    current_data = allData[dataCounter]
    dataCounter = (dataCounter + 1) % len(allData)


    batch_size = 128
    embedding_size = 64   # Dimension of the embedding vector.
    skip_window = 4       # How many words to consider left and right.
    num_skips = 8         # How many times to reuse an input to generate a label.
    num_sampled = 16      # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

      # Ops and variables pinned to the CPU because of missing GPU implementation
      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      # Explanation of the meaning of NCE loss:
      #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
      loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size))

      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
      #optimizer = tf.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)
      #optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)

      # Add variable initializer.
      init = tf.global_variables_initializer()

      saver = tf.train.Saver()

    # Step 5: Begin training.
    num_steps = 1000001

    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print('Initialized')

      average_loss = 0
      for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 100000 == 0:
          sim = similarity.eval()
          for i in xrange(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in xrange(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
      final_embeddings = normalized_embeddings.eval()

      save_path = saver.save(session, "./model.ckpt")
      print("Model saved in path: %s" % save_path)

      try:
        # pylint: disable=g-import-not-at-top

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 200
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, 'tsne.png')

      except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)


if __name__=="__main__":
    main()
