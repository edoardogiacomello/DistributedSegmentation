import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
tf.test.is_gpu_available()

BASE_MODEL_PATH = 'models/model_base.h5'
MODEL_OUT_FOLDER = 'models/'
ID_TO_LABEL = {16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse'}
LABEL_TO_ID = {'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19}
CHANNEL_ORDER = [0, 16, 17, 18, 19] # Order of channels in output segmentation and corresponding dataset labels
CHANNEL_NAMES = [ID_TO_LABEL[i] if i!=0 else 'other' for i in CHANNEL_ORDER]
ALL_LABELS = list(LABEL_TO_ID.keys())
ds_csv_paths = {dset: {label: 'datasets/coco_animals_{}_{}.csv'.format(dset, label) for label in ALL_LABELS} for dset in ['train', 'validation', 'test']}

def load_dataset(path, size=224, batch_size=32):
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
        for lid in CHANNEL_ORDER:
            # Creating 5 masks out of the index labels
            segs.append(tf.cast(tf.equal(seg, lid), tf.float32))
        seg = tf.concat(segs, axis=-1)
        return png, seg
    dataset = tf.data.experimental.CsvDataset(path, [tf.string, tf.string, tf.string, tf.int32], header=True)
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(parse_sample)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def show_one_sample(dataset_label):
    for png, seg in load_dataset(ds_csv_paths['train'][dataset_label]):
        png = png.numpy()[0,...]
        seg = seg.numpy()[0,...]
        break
    plt.figure(figsize=(18, 6))
    plt.subplot(1, seg.shape[-1]+1, 1)
    plt.axis('off')
    io.imshow(png)
    for i in range(seg.shape[-1]):
        plt.subplot(1, seg.shape[-1]+1, i+2)
        plt.axis('off')
        io.imshow(seg[...,i])
        
def train_model(label, input_model=BASE_MODEL_PATH, output_folder=MODEL_OUT_FOLDER, epochs=1000, batch_size=32, seed=123456780):
    tf.random.set_seed(seed)
    # Load model
    base_model = tk.models.load_model(BASE_MODEL_PATH)
    base_model.compile('adam', loss='mae', metrics=[tf.keras.metrics.Accuracy(), 
                                                    tf.keras.metrics.BinaryAccuracy(), 
                                                    tf.keras.metrics.FalsePositives(), 
                                                    tf.keras.metrics.FalseNegatives(), 
                                                    tf.keras.metrics.Precision(), 
                                                    tf.keras.metrics.Recall()])
    save_callback=tk.callbacks.ModelCheckpoint('models/model_'+label+'-{epoch:03d}-{val_binary_accuracy:.2f}.h5', period=10)
    
    # Train the model
    return base_model.fit(load_dataset(ds_csv_paths['train'][label]), epochs=epochs, validation_data=load_dataset(ds_csv_paths['validation'][label]), callbacks=[save_callback])
    
    
    
def predict(model, dataset_label):
    for png, seg in load_dataset(ds_csv_paths['validation'][dataset_label], batch_size=1):
        out = model.predict(png)[0,...]
        png = png.numpy()[0,...]
        seg = seg.numpy()[0,...]
        
        
        plt.figure(figsize=(18, 6))
        plt.subplot(2, seg.shape[-1]+1, 1)
        plt.axis('off')
        io.imshow(png)
        for i in range(seg.shape[-1]):
            plt.subplot(2, seg.shape[-1]+1, i+2)
            plt.axis('off')
            plt.title(CHANNEL_NAMES[i])
            io.imshow(seg[...,i])
        for i in range(out.shape[-1]):
            plt.subplot(2, out.shape[-1]+1, seg.shape[-1]+1+i+2)
            plt.axis('off')
            plt.title(CHANNEL_NAMES[i])
            io.imshow(out[...,i])
        yield plt
    
    
if __name__=="__main__":
    for target_label in ['horse', 'cat']:
        history = train_model(target_label, epochs=500)
        hist = pd.DataFrame(history.history)
        hist.to_csv('models/history_{}.csv'.format(target_label))
    