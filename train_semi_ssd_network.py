# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a SSD model using a given dataset."""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import tf_utils

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC'

# =========================================================================== #
# Semi-supervised learning flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'T1', 20, 'T1 parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'T2', 120, 'T2 parameter in the loss function.')

# =========================================================================== #
# SSD Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')
# =========================================================================== #
# SSD evaluation Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_integer(
    'select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer(
    'keep_top_k', 200, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float(
    'matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_integer(
    'eval_resize', 4, 'Image resizing: None / CENTRAL_CROP / PAD_AND_RESIZE / WARP_RESIZE.')
tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size.')
tf.app.flags.DEFINE_boolean(
    'remove_difficult', True, 'Remove difficult objects from evaluation.')

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'labeled_dataset_name', 'mani', 'The name of the labeled dataset to load.')
tf.app.flags.DEFINE_string(
    'unlabeled_dataset_name', 'mani_unlabeled', 'The name of the unlabeled dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 1, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'labeled_dir', None, 'The directory where the labeled dataset files are stored.')
tf.app.flags.DEFINE_string(
    'unlabeled_dir', None, 'The directory where the unlabeled dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'semi_ssd_300_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    # if not FLAGS.dataset_dir:
    #     raise ValueError('You must supply the dataset directory with --dataset_dir')
    # if not FLAGS.dataset_dir:
    #     raise ValueError('You must supply the unlabeled set directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # Config model_deploy. Keep TF Slim Models structure.
        # Useful if want to need multiple GPUs and/or servers in the future.
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0)
        # Create global_step.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        # Select the dataset.
        labeled_dataset = dataset_factory.get_dataset(
            FLAGS.labeled_dataset_name, FLAGS.dataset_split_name, FLAGS.labeled_dir)
        unlabeled_dataset = dataset_factory.get_dataset(
            FLAGS.unlabeled_dataset_name, FLAGS.dataset_split_name, FLAGS.unlabeled_dir)

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=True)

        tf_utils.print_configuration(FLAGS.__flags, ssd_params,
                                     labeled_dataset.data_sources, FLAGS.train_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.device(deploy_config.inputs_device()):
            with tf.name_scope(FLAGS.labeled_dataset_name + '_data_provider'):
                labeled_provider = slim.dataset_data_provider.DatasetDataProvider(
                    labeled_dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size,
                    shuffle=True)

                unlabeled_provider = slim.dataset_data_provider.DatasetDataProvider(
                    unlabeled_dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size,
                    shuffle=True)

            step_per_epoch = labeled_provider.num_samples() // FLAGS.batch_size
            # Get labeled data for SSD network: image, labels, bboxes.
            labeled_data = dict()
            labeled_data['image'], labeled_data['labels'], labeled_data['bboxes'] = \
                labeled_provider.get(['image', 'object/label', 'object/bbox'])
            # Get unlabeled data for SSD network: image
            unlabeled_data = dict()
            unlabeled_data['image'] = unlabeled_provider.get(['image'])[0]
            # Pre-processing image, labels and bboxes.
            labeled_data['image'], labeled_data['labels'], labeled_data['bboxes'] = \
                image_preprocessing_fn(labeled_data['image'], labeled_data['labels'], labeled_data['bboxes'],
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT)

            unlabeled_data['image'], _, _ = \
                image_preprocessing_fn(unlabeled_data['image'], None, None,
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT)
            # Encode groundtruth labels and bboxes.
            labeled_data['classes'], labeled_data['localisations'], labeled_data['scores'] = \
                ssd_net.bboxes_encode(labeled_data['labels'], labeled_data['bboxes'], ssd_anchors)

            labeled_batch_shape = [1] + [len(ssd_anchors)] * 3
            unlabeled_batch_shape = [1]

            # Training batches and queue.
            r_labeled = tf.train.batch(
                tf_utils.reshape_list([labeled_data['image'], labeled_data['classes'], labeled_data['localisations'], labeled_data['scores']]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
            labeled_batch = dict()
            labeled_batch['image'], labeled_batch['classes'], labeled_batch['localisations'], labeled_batch['scores'] = \
                tf_utils.reshape_list(r_labeled, labeled_batch_shape)

            r_unlabeled = tf.train.batch(
                [unlabeled_data['image']],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
            unlabeled_batch = dict()
            unlabeled_batch['image'] = r_unlabeled

            # Intermediate queueing: unique batch computation pipeline for all
            # GPUs running the training.
            labeled_queue = slim.prefetch_queue.prefetch_queue(
                tf_utils.reshape_list([labeled_batch['image'], labeled_batch['classes'], labeled_batch['localisations'], labeled_batch['scores']]),
                capacity=2 * deploy_config.num_clones)

            unlabeled_queue = slim.prefetch_queue.prefetch_queue(
                [unlabeled_batch['image']],
                capacity=2 * deploy_config.num_clones)
        # =================================================================== #
        # Define the model running on every GPU.
        # =================================================================== #
        def clone_fn(labeled_queue, unlabeled_queue):
            """Allows data parallelism by creating multiple
            clones of network_fn."""
            # Dequeue batch.
            labeled_batch = dict()
            labeled_batch['image'], labeled_batch['classes'], labeled_batch['localisations'], labeled_batch['scores'] = \
                tf_utils.reshape_list(labeled_queue.dequeue(), labeled_batch_shape)
            unlabeled_batch = dict()
            unlabeled_batch['image'] = unlabeled_queue.dequeue()

            # Construct SSD network.
            arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                          data_format=DATA_FORMAT)
            with slim.arg_scope(arg_scope):
                labeled_preds = dict()
                labeled_preds['predictions'], labeled_preds['localisations'], labeled_preds['logits'], labeled_preds['end_points'] = \
                    ssd_net.net(labeled_batch['image'], is_training=True)

                unlabeled_preds = dict()
                unlabeled_preds['predictions'], unlabeled_preds['localisations'], unlabeled_preds['logits'], unlabeled_preds['end_points'] = \
                    ssd_net.net(unlabeled_batch['image'], is_training=True, reuse=True)

                unlabeled_pseudo = dict()
                unlabeled_pseudo['predictions'], unlabeled_pseudo['localisations'], unlabeled_pseudo['logits'], unlabeled_pseudo['end_points'] = \
                    ssd_net.net(unlabeled_batch['image'], is_training=False, reuse=True)

                unlabeled_pseudo['localisations'] = ssd_net.bboxes_decode(unlabeled_pseudo['localisations'], ssd_anchors)
                unlabeled_pseudo['labels'], unlabeled_pseudo['bboxes'] = \
                    ssd_net.detected_bboxes(unlabeled_pseudo['predictions'], unlabeled_pseudo['localisations'],
                                            select_threshold=FLAGS.select_threshold,
                                            nms_threshold=FLAGS.nms_threshold,
                                            clipping_bbox=None,
                                            top_k=FLAGS.select_top_k,
                                            keep_top_k=FLAGS.keep_top_k)
                # unlabeled_pseudo['labels'] = tf.cast(tf.reshape(unlabeled_pseudo['labels'][0], shape=[-1]), dtype=tf.int64)
                # unlabeled_pseudo['bboxes'] = tf.reshape(unlabeled_pseudo['bboxes'][0], shape=[-1, 4])

                unlabeled_pseudo['labels'] = tf.cast(unlabeled_pseudo['labels'][0], dtype=tf.int64)
                unlabeled_pseudo['bboxes'] = unlabeled_pseudo['bboxes'][0]

                # unlabeled_pseudo['classes'], unlabeled_pseudo['localisations'], unlabeled_pseudo['scores'] = \
                #     ssd_net.bboxes_encode(unlabeled_pseudo['labels'], unlabeled_pseudo['bboxes'], ssd_anchors)
                unlabeled_pseudo['classes'], unlabeled_pseudo['localisations'], unlabeled_pseudo['scores'] = tf.map_fn(lambda items: ssd_net.bboxes_encode(items[0], items[1], ssd_anchors),
                                [unlabeled_pseudo['labels'], unlabeled_pseudo['bboxes']],
                                dtype=([tf.int64]*6, [tf.float32]*6, [tf.float32]*6))

            # Add loss function.
            ssd_net.semi_losses(labeled_preds,
                                labeled_batch,
                                unlabeled_preds,
                                unlabeled_pseudo,
                                global_step,
                                step_per_epoch,
                                T1=FLAGS.T1,
                                T2=FLAGS.T2,
                                match_threshold=FLAGS.match_threshold,
                                negative_ratio=FLAGS.negative_ratio,
                                alpha=FLAGS.loss_alpha,
                                label_smoothing=FLAGS.label_smoothing)
            return labeled_preds['end_points']

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # =================================================================== #
        # Add summaries from first clone.
        # =================================================================== #
        clones = model_deploy.create_clones(deploy_config, clone_fn, [labeled_queue, unlabeled_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))
        # Add summaries for losses and extra losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        for loss in tf.get_collection('labeled_losses', first_clone_scope):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        for loss in tf.get_collection('unlabeled_losses', first_clone_scope):
            summaries.add(tf.summary.scalar(loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        # =================================================================== #
        # Configure the moving averages.
        # =================================================================== #
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        # =================================================================== #
        # Configure the optimization procedure.
        # =================================================================== #
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf_utils.configure_learning_rate(FLAGS,
                                                             labeled_provider.num_samples(),
                                                             global_step)
            optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = tf_utils.get_variables_to_train(FLAGS)

        # and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                          name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # =================================================================== #
        # Kicks off the training.
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options=gpu_options)
        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)

        # def train_step_fn(sess, train_op, global_step, train_step_kwargs):
        #
        #     step = sess.run(global_step)
        #     if step % step_per_epoch == 0:
        #         sess.run()
        #
        #     return slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            # train_step_fn=train_step_fn,
            master='',
            is_chief=True,
            init_fn=tf_utils.get_init_fn(FLAGS),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            saver=saver,
            save_interval_secs=FLAGS.save_interval_secs,
            session_config=config,
            sync_optimizer=None)


if __name__ == '__main__':
    tf.app.run()
