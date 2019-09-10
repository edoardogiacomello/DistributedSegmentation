import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os

# Setting allow_growth for gpu
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found, model running on CPU")

class SegmentationModel():
    def __init__(self, label, size=224, threshold=0.5):
        self.target_label = label
        self.BASE_MODEL_PATH = 'models/model_base.h5'
        self.MODEL_OUT_FOLDER = 'models/{}/'.format(label)
        os.makedirs(self.MODEL_OUT_FOLDER, exist_ok=True)
        self.ID_TO_LABEL = {16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse'}
        self.LABEL_TO_ID = {'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19}
        self.CHANNEL_ORDER = [0, 16, 17, 18, 19] # Order of channels in output segmentation and corresponding dataset labels
        self.CHANNEL_NAMES = [self.ID_TO_LABEL[i] if i!=0 else 'other' for i in self.CHANNEL_ORDER]
        self.ALL_LABELS = list(self.LABEL_TO_ID.keys())
        self.ds_csv_paths = {dset: {label: 'datasets/coco_animals_{}_{}.csv'.format(dset, label) for label in self.ALL_LABELS} for dset in ['train', 'validation', 'test']}
        self.n_labels = len(self.CHANNEL_ORDER)
        self.size = size
        self.threshold = threshold
        self.epoch=0
        
    
    def load_dataset(self, path, size=224, batch_size=32, filter_expr=None):
        def parse_sample(png_path, seg_path, lab_name, lab_value):
            resize = tf.image.resize_image_with_pad if tf.__version__.startswith('1.') else tf.image.resize_with_pad
            png_raw = tf.io.read_file(png_path)
            png = tf.image.decode_png(png_raw, channels=3)
            png = resize(png, size, size)
            png = preprocess_input(tf.cast(png, tf.float32))
            seg_raw = tf.io.read_file(seg_path)
            seg = tf.image.decode_png(seg_raw, channels=1)
            seg = resize(seg, size, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            segs = []
            for lid in self.CHANNEL_ORDER:
                # Creating 5 masks out of the index labels
                segs.append(tf.cast(tf.equal(seg, lid), tf.float32))
            seg = tf.concat(segs, axis=-1)
            return png, seg
        dataset = tf.data.experimental.CsvDataset(path, [tf.string, tf.string, tf.string, tf.int32], header=True)
        if filter_expr:
            dataset = dataset.filter(filter_expr)
        dataset = dataset.shuffle(1000)
        dataset = dataset.map(parse_sample)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset
    
    def load_test_dataset(self, path, size=224, batch_size=1, filter_expr=None):
        ''' Loads the test dataset. Default parameter is 1 and the dataset is not shuffled. The iterator also returns image paths'''
        def parse_sample(png_path, seg_path, lab_name, lab_value):
            resize = tf.image.resize_image_with_pad if tf.__version__.startswith('1.') else tf.image.resize_with_pad
            png_raw = tf.io.read_file(png_path)
            png = tf.image.decode_png(png_raw, channels=3)
            png = resize(png, size, size)
            png = preprocess_input(tf.cast(png, tf.float32))
            seg_raw = tf.io.read_file(seg_path)
            seg = tf.image.decode_png(seg_raw, channels=1)
            seg = resize(seg, size, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            segs = []
            for lid in self.CHANNEL_ORDER:
                # Creating 5 masks out of the index labels
                segs.append(tf.cast(tf.equal(seg, lid), tf.float32))
            seg = tf.concat(segs, axis=-1)
            return png_path, seg_path, png, seg
        dataset = tf.data.experimental.CsvDataset(path, [tf.string, tf.string, tf.string, tf.int32], header=True)
        if filter_expr:
            dataset = dataset.filter(filter_expr)
        dataset = dataset.map(parse_sample)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset
    
    def load_base_network(self):
        self.model = tk.models.load_model(self.BASE_MODEL_PATH)
    
    def load_finetuned_network(self, epoch):
        self.model = tk.models.load_model(self.MODEL_OUT_FOLDER+'model_ep{}.h5'.format(epoch))
        self.epoch = epoch
        print("Loaded model for label {} at epoch {}".format(self.target_label, self.epoch))
    
    def define_metric_accumulators(self):
        self.train_loss = tf.metrics.Mean()
        self.train_IoU = {label: tf.metrics.MeanIoU(num_classes=2) for label in self.ALL_LABELS}        
        self.valid_loss = tf.metrics.Mean()
        self.valid_IoU = {label: tf.metrics.MeanIoU(num_classes=2) for label in self.ALL_LABELS}

    @tf.function
    def train_step(self, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.model(x,training=True)
            loss = self.loss(y_true, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    @tf.function
    def validation_step(self, x, y_true):
        y_pred = self.model(x)
        loss = self.loss(y_true, y_pred)
        return loss

    
    def train(self, epochs=1000, batch_size=32, seed=123456780):
        self.train_summary_writer = tf.summary.create_file_writer(self.MODEL_OUT_FOLDER+'train/')
        self.valid_summary_writer = tf.summary.create_file_writer(self.MODEL_OUT_FOLDER+'validation/')
        
        tf.random.set_seed(seed)
        
        self.load_base_network()
        self.define_metric_accumulators()
        
        self.loss = tk.losses.CategoricalCrossentropy()        
        self.optimizer = tk.optimizers.Adam()
        progbar = tk.utils.Progbar(None)
        for e in range(self.epochs+1,  epochs):
            i = 0
            # Train Step
            for x, y in self.load_dataset(self.ds_csv_paths['train'][self.target_label], batch_size=batch_size, size=self.size):
                train_step_loss = self.train_step(x, y)
                self.train_loss.update_state(train_step_loss)
                i += 1; progbar.update(i);
            
            i = 0
            # Validation Step
            for x, y in self.load_dataset(self.ds_csv_paths['validation'][self.target_label], batch_size=batch_size, size=self.size):
                valid_step_loss = self.validation_step(x, y)
                self.valid_loss.update_state(valid_step_loss)
                i += 1; progbar.update(i);
        
            # Calculate train/validation metrics for each label separately
            
            for filter_label in self.ALL_LABELS:
                # Filtering only data containing the target label
                data_filter = lambda png, seg, label, label_id, target_lab=filter_label: tf.equal(label, self.target_label)
                i = 0
                for x, y_true in self.load_dataset(self.ds_csv_paths['train'][self.target_label], batch_size=batch_size, size=self.size, filter_expr=data_filter):
                    y_pred = self.model(x)
                    self.train_IoU[filter_label].update_state(y_true, tf.cast(y_pred > self.threshold, dtype=tf.float32), sample_weight=tf.tile(tf.one_hot(self.CHANNEL_NAMES.index(filter_label), depth=len(self.CHANNEL_NAMES))[tf.newaxis, tf.newaxis, tf.newaxis, ...], [y_pred.shape[0], self.size,self.size,1]))
                    i += 1; progbar.update(i);
                i = 0
                for x, y_true in self.load_dataset(self.ds_csv_paths['validation'][self.target_label], batch_size=batch_size, size=self.size, filter_expr=data_filter):
                    y_pred = self.model(x)
                    self.valid_IoU[filter_label].update_state(y_true,  tf.cast(y_pred > self.threshold, dtype=tf.float32), sample_weight=tf.tile(tf.one_hot(self.CHANNEL_NAMES.index(filter_label), depth=len(self.CHANNEL_NAMES))[tf.newaxis, tf.newaxis, tf.newaxis, ...], [y_pred.shape[0], self.size,self.size,1]))
                    i += 1; progbar.update(i);

            # Visualization
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=e)
                self.train_loss.reset_states()
                for l in self.train_IoU.keys():
                    tf.summary.scalar('IOU_{}'.format(l), self.train_IoU[l].result(), step=e)
                    self.train_IoU[l].reset_states()
                    
            with self.valid_summary_writer.as_default():
                tf.summary.scalar('loss', self.valid_loss.result(), step=e)
                self.valid_loss.reset_states()
                for l in self.train_IoU.keys():
                    tf.summary.scalar('IOU_{}'.format(l), self.valid_IoU[l].result(), step=e)
                    self.valid_IoU[l].reset_states()
            # Saving
            tk.models.save_model(self.model, self.MODEL_OUT_FOLDER+'model_ep{}.h5'.format(e))
            self.epochs = e

    def predict(self, x):
        return self.model(x)
    
    def predict_raw(self, raw_file):
        x = tf.image.decode_png(raw_file, channels=3)
        x = resize(png, size, size)
        x = preprocess_input(tf.cast(png, tf.float32))
        return self.model(x)
        
        
