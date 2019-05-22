# TX2_tracking
The dataset should be preprocessed into TFRecord format.
  - convertToTFRecords.py can help this preprocessing.
  - The folder should contain the images and the corresponding XML files.
  - In convertToTFRecords.py, you can specifiy the ration of training dataset and validation dataset.
  - You have to define the labels for your images.
 
 Training
   - tf_image_classifier.py is used for training.
   - The default size of the image is 360x640.
   - mse_loss = tf.losses.mean_squared_error(labels=Y_BBOX, predictions=Y_bbox, weights=1e-3) which has the weight 1e-3
   - cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y_, logits=Y_class, weights=1e-2)) which has the weight 1e-2
   - total_loss = mse_loss + cross_entropy + reg_loss
   - Three parameters are used to specify the architecture of the inducing neural network
       + K = 98 # number of classes
       + G = 512 # number of grid cells
       + P = 4  # four parameters of the bounding boxes
   - The image is read from the TFRecord so it is natural to use the image processing API provided by TensorFlow.
       + Ex. tf.subtract()
   
Inference
  - tf_image_inference.py
  - Load the model and do the inference.
