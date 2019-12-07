import tensorflow as tf


class Logger(object):
    # print(tf.__version__)
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        #self.writer = tf.summary.FileWriter(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)
        #with summary_writer.as_default():
        #    tf.summary.scalar('loss', 0.1, step=42)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.summary(value=[tf.summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        summary = tf.summary(value=[tf.summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        self.writer.add_summary(summary, step)
