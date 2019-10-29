import tensorflow as tf

"""https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables/43601894#43601894"""
def initialize_uninitialized(sess):
  global_vars          = tf.global_variables()
  is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
  not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

  if len(not_initialized_vars):
      sess.run(tf.variables_initializer(not_initialized_vars))