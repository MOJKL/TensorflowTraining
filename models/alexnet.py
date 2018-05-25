import tensorflow as tf

# Network Parameters
dropout = 0.5 # Dropout, probability to drop a unit

# Create the neural network
def alex_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 227, 227, 3])

        # Convolution Layer with 96 filters and a kernel size of 11
        conv1 = tf.layers.conv2d(
                inputs=x,
                filters=96,
                kernel_size=[11,11],
                strides=4,
                padding="valid",
                activation=tf.nn.relu)
        
        # Normalization 1                                   
        norm1 = tf.contrib.layers.layer_norm(conv1)
        
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 3
        pool1 = tf.layers.max_pooling2d(
                inputs=norm1, 
                pool_size=[3,3],
                strides=2)

        # Convolution Layer with 256 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=256,
                kernel_size=[5,5],
                strides = 1,
                padding="same",
                activation=tf.nn.relu)
        
        # Normalization 2
        norm2 = tf.contrib.layers.layer_norm(conv2)
        
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool2 = tf.layers.max_pooling2d(
                inputs=norm2,
                pool_size=[3,3],
                strides=2)
        
        # Convolution Layer with 384 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=384,
                kernel_size=[3,3],
                strides = 1,
                padding="same",
                activation=tf.nn.relu)
        
        # Convolution Layer with 384 filters and a kernel size of 3
        conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=384,
                kernel_size=[3,3],
                strides = 1,
                padding="same",
                activation=tf.nn.relu)
    
        # Convolution Layer with 256 filters and a kernel size of 3
        conv5 = tf.layers.conv2d(
                inputs=conv4,
                filters=256,
                kernel_size=[3,3],
                strides = 1,
                padding="same",
                activation=tf.nn.relu)
         
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 3
        pool3 = tf.layers.max_pooling2d(
                inputs=conv5,
                pool_size=[3,3],
                strides=2)
        
        # Flatten
        flat1 = tf.reshape(pool3,[-1, 9216])
        
        # Fully connected layer 1
        fc1 = tf.layers.dense(
                inputs=flat1,
                units=4096,
                activation=tf.nn.relu)
            
        # Dropout if training
        dropout1 = tf.layers.dropout(
                inputs=fc1,
                rate=dropout,
                training=is_training)
        
        # Fully connected layer 2
        fc2 = tf.layers.dense(
                inputs=dropout1,
                units=4096,
                activation=tf.nn.relu)
    
        # Dropout if training                                  
        dropout2 = tf.layers.dropout(
                inputs=fc2,
                rate=dropout,
                training=is_training)
        
        # Output layer, class prediction                         
        out = tf.layers.dense(inputs=dropout2, units=n_classes)

    return out
