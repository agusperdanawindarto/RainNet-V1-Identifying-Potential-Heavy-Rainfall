import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, kernel_size=3, filter_num=32, pool_size=2, strides=1,
    **kwargs):

    super(Encoder, self).__init__()
    self.kernel_size = kernel_size
    self.filter_num = filter_num
    self.pool_size = pool_size
    self.strides = strides

  def build(self, input_shape):
    self.conv2D_enc_1 = self.conv2D_fixed_padding(
      self.filter_num, self.kernel_size, self.strides)
    self.conv2D_enc_2 = self.conv2D_fixed_padding(
      self.filter_num, self.kernel_size, self.strides)
    self.conv2D_enc_3 = self.conv2D_fixed_padding(
      self.filter_num, 1, self.strides)

    self.batch_norm_1 = self.batch_norm()
    self.batch_norm_2 = self.batch_norm()
    self.batch_norm_3 = self.batch_norm()

    self.activation_1 = tf.keras.layers.ReLU()
    self.activation_2 = tf.keras.layers.ReLU()
    self.activation_3 = tf.keras.layers.ReLU()

    self.maxPool = tf.keras.layers.MaxPool2D(
      pool_size=self.pool_size)


  def call(self, inputs):
    x = self.conv2D_enc_1(inputs)
    x = self.batch_norm_1(x)
    x = self.activation_1(x)

    x = self.conv2D_enc_2(x)
    x = self.batch_norm_2(x)
    x = self.activation_2(x)

    y = self.conv2D_enc_3(inputs)
    y = self.batch_norm_3(y)

    x = self.activation_3(x + y)
    return self.maxPool(x), x


  def get_config(self):
    config = super().get_config().copy()
    config.update({
      "kernel_size": self.kernel_size,
      "filter_num": self.filter_num,
      "pool_size": self.pool_size,
      "strides": self.strides})

    return config


  @staticmethod
  def batch_norm():
    """Performs a batch normalization using a standard set of parameters."""
    return tf.keras.layers.BatchNormalization(
      momentum=0.997, epsilon=1e-5)

  @staticmethod
  def fixed_padding(kernel_size):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1)
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    return tf.keras.layers.ZeroPadding2D(
      padding=(([pad_beg, pad_end]), (pad_beg, pad_end)))

  @staticmethod
  def conv2D_fixed_padding(filters, kernel_size, strides):
    """Strided 2-D convolution with explicit padding.""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    """
    if strides > 1:
      return tf.keras.Sequential([
        Encoder.fixed_padding(kernel_size),
        tf.keras.layers.Conv2D(
          filters=filters, kernel_size=kernel_size, strides=strides,
          padding=("SAME" if strides == 1 else "VALID"), use_bias=False,
          kernel_initializer=tf.keras.initializers.VarianceScale())
        ])
    else:
      return tf.keras.layers.Conv2D(
          filters=filters, kernel_size=kernel_size, strides=strides,
          padding=("SAME" if strides == 1 else "VALID"), use_bias=False,
          kernel_initializer=tf.keras.initializers.VarianceScaling())

class Bridge(tf.keras.layers.Layer):
  def __init__(self, kernel_size=3, filter_num=32, dropout_rate=0.5,
    strides=1, **kwargs):

    super(Bridge, self).__init__()
    self.kernel_size = kernel_size
    self.filter_num = filter_num
    self.dropout_rate = dropout_rate
    self.strides = strides

  def build(self, input_shape):
    self.conv2D_bridge_1 = self.conv2D_fixed_padding(
      self.filter_num, self.kernel_size, self.strides)
    self.conv2D_bridge_2 = self.conv2D_fixed_padding(
      self.filter_num, self.kernel_size, self.strides)
    self.conv2D_bridge_3 = self.conv2D_fixed_padding(
      self.filter_num, 1, self.strides)

    self.batch_norm_1 = self.batch_norm()
    self.batch_norm_2 = self.batch_norm()
    self.batch_norm_3 = self.batch_norm()

    self.activation_1 = tf.keras.layers.ReLU()
    self.activation_2 = tf.keras.layers.ReLU()
    self.activation_3 = tf.keras.layers.ReLU()

    self.dropout = tf.keras.layers.Dropout(self.dropout_rate)


  def call(self, inputs):
    x = self.conv2D_bridge_1(inputs)
    x = self.batch_norm_1(x)
    x = self.activation_1(x)

    x = self.conv2D_bridge_2(x)
    x = self.batch_norm_2(x)
    x = self.activation_2(x)

    y = self.conv2D_bridge_3(inputs)
    y = self.batch_norm_3(y)

    x = self.activation_3(x + y)
    return self.dropout(x)


  def get_config(self):
    config = super().get_config().copy()
    config.update({
      "kernel_size": self.kernel_size,
      "filter_num": self.filter_num,
      "dropout_rate": self.dropout_rate,
      "strides": self.strides})

    return config


  @staticmethod
  def batch_norm():
    """Performs a batch normalization using a standard set of parameters."""
    return tf.keras.layers.BatchNormalization(
      momentum=0.997, epsilon=1e-5)

  @staticmethod
  def fixed_padding(kernel_size):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1)
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    return tf.keras.layers.ZeroPadding2D(
      padding=(([pad_beg, pad_end]), (pad_beg, pad_end)))

  @staticmethod
  def conv2D_fixed_padding(filters, kernel_size, strides):
    """Strided 2-D convolution with explicit padding.""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    """
    if strides > 1:
      return tf.keras.Sequential([
        Bridge.fixed_padding(kernel_size),
        tf.keras.layers.Conv2D(
          filters=filters, kernel_size=kernel_size, strides=strides,
          padding=("SAME" if strides == 1 else "VALID"), use_bias=False,
          kernel_initializer=tf.keras.initializers.VarianceScale())
        ])
    else:
      return tf.keras.layers.Conv2D(
          filters=filters, kernel_size=kernel_size, strides=strides,
          padding=("SAME" if strides == 1 else "VALID"), use_bias=False,
          kernel_initializer=tf.keras.initializers.VarianceScaling())

class Decoder(tf.keras.layers.Layer):
  def __init__(self, kernel_size=3, filter_num=32, pool_size=2, strides=1, **kwargs):
    super(Decoder, self).__init__()
    self.kernel_size = kernel_size
    self.filter_num = filter_num
    self.pool_size = pool_size
    self.strides = strides

  def build(self, input_shape):
    self.conv2D_transpose = tf.keras.layers.Conv2DTranspose(
      kernel_size=self.kernel_size, filters=self.filter_num,
      strides=2, padding="SAME")
    self.concatenate = tf.keras.layers.Concatenate(axis=3)

    self.conv2D_dec_1 = self.conv2D_fixed_padding(
      self.filter_num, self.kernel_size, self.strides)
    self.conv2D_dec_2 = self.conv2D_fixed_padding(
      self.filter_num, self.kernel_size, self.strides)
    self.conv2D_dec_3 = self.conv2D_fixed_padding(
      self.filter_num, 1, self.strides)

    self.batch_norm_1 = self.batch_norm()
    self.batch_norm_2 = self.batch_norm()
    self.batch_norm_3 = self.batch_norm()

    self.activation_1 = tf.keras.layers.ReLU()
    self.activation_2 = tf.keras.layers.ReLU()
    self.activation_3 = tf.keras.layers.ReLU()

    self.maxPool = tf.keras.layers.MaxPool2D(
      pool_size=self.pool_size)


  def call(self, inputs, skip):
    x = self.conv2D_transpose(inputs)
    x = self.activation_1(x)
    x = self.concatenate([x, skip])

    # projection
    y = self.conv2D_dec_3(x)
    y = self.batch_norm_3(y)

    x = self.conv2D_dec_1(x)
    x = self.batch_norm_1(x)
    x = self.activation_2(x)

    x = self.conv2D_dec_2(x)
    x = self.batch_norm_2(x)

    return self.activation_3(x + y)


  def get_config(self):
    config = super().get_config().copy()
    config.update({
      "kernel_size": self.kernel_size,
      "filter_num": self.filter_num,
      "pool_size": self.pool_size,
      "strides": self.strides})

    return config

  @staticmethod
  def batch_norm():
    """Performs a batch normalization using a standard set of parameters."""
    return tf.keras.layers.BatchNormalization(
      momentum=0.997, epsilon=1e-5)

  @staticmethod
  def fixed_padding(kernel_size):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1)
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    return tf.keras.layers.ZeroPadding2D(
      padding=(([pad_beg, pad_end]), (pad_beg, pad_end)))

  @staticmethod
  def conv2D_fixed_padding(filters, kernel_size, strides):
    """Strided 2-D convolution with explicit padding.""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    """
    if strides > 1:
      return tf.keras.Sequential([
        Decoder.fixed_padding(kernel_size),
        tf.keras.layers.Conv2D(
          filters=filters, kernel_size=kernel_size, strides=strides,
          padding=("SAME" if strides == 1 else "VALID"), use_bias=False,
          kernel_initializer=tf.keras.initializers.VarianceScale())
        ])
    else:
      return tf.keras.layers.Conv2D(
          filters=filters, kernel_size=kernel_size, strides=strides,
          padding=("SAME" if strides == 1 else "VALID"), use_bias=False,
          kernel_initializer=tf.keras.initializers.VarianceScaling())

def dice_coeff_multi(y_true, y_pred):
  SMOOTH = 1.    # to avoid division by 0

  # from one-hot encoding to labels
  y_true = tf.argmax(y_true, axis=-1)
  y_pred = tf.argmax(y_pred, axis=-1)

  y_true_f = tf.keras.layers.Flatten()(y_true)
  y_pred_f = tf.keras.layers.Flatten()(y_pred)

  intersection = tf.reduce_sum(
    tf.cast( tf.math.not_equal(y_true_f*y_pred_f, 0), tf.float32))

  return (2. * intersection + SMOOTH) \
    / (tf.reduce_sum(
        tf.cast(tf.math.not_equal(y_true_f, 0), tf.float32))
        + tf.reduce_sum(
            tf.cast(tf.math.not_equal(y_pred_f, 0), tf.float32)) + SMOOTH)

def mean_iou(y_true, y_pred, num_of_targets):
  # from one-hot encoding to labels
  y_true = tf.argmax(y_true, axis=-1)
  y_pred = tf.argmax(y_pred, axis=-1)
  # print(y_true.shape)
  # print(y_pred.shape)

  y_true_f = tf.keras.layers.Flatten()(y_true).numpy()
  y_pred_f = tf.keras.layers.Flatten()(y_pred).numpy()

  y_true_f = y_true_f.astype(int)
  y_pred_f = y_pred_f.astype(int)
  
  # print(f"{np.unique(y_true_f)}")
  # print(f"{np.unique(y_pred_f)}")
  m_mean_iou_result = 0
  for label in range(1, num_of_targets+1):
    m_iou = tf.keras.metrics.IoU(num_classes=num_of_targets+1, target_class_ids=[label])
    m_iou.update_state(y_true_f, y_pred_f)
    m_mean_iou_result += m_iou.result().numpy()
  
  return m_mean_iou_result/num_of_targets

def get_all_saved_models(reload_filename_arr, model_name, filter_seq_arr, saved_dir):
  reload_saved_models_arr = {k: None for k in filter_seq_arr}
  for key, reload_filename in zip(filter_seq_arr, reload_filename_arr):
    print(f"[LOAD]: {reload_filename}")

    if model_name == "RainNet":
      reload_saved_models_arr[key] = tf.keras.models.load_model(
          saved_dir / f"{reload_filename}.h5",
          custom_objects={
            "dice_coeff_multi": dice_coeff_multi,
            "Encoder": Encoder,
            "Bridge": Bridge,
            "Decoder": Decoder})
    elif model_name == "UNet":
      reload_saved_models_arr[key] = tf.keras.models.load_model(
          saved_dir / f"{reload_filename}.h5",
          custom_objects={
            "dice_coeff_multi": dice_coeff_multi})

  return reload_saved_models_arr


