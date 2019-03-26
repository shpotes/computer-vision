import random
import pickle
import pathlib
import tensorflow as tf

def load_and_preprocess(path, label, num_classes=1000):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (227, 227))
    image /= 255.0
    return image, tf.one_hot(label, num_classes)

def read_image_dataset(folder, imagenet=True):
    data_root = pathlib.Path(folder)
    all_image_paths = [str(path) for path in data_root.glob('*/*')]
    N = len(all_image_paths)
    random.shuffle(all_image_paths)
    
    if imagenet:
        all_image_labels = [int(pathlib.Path(path).parent.name) - 1 
                    for path in all_image_paths]
        label_map = pickle.load(open("classes.pkl", "rb" ))
    
    else:
        pass
    
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    return ds.map(load_and_preprocess), N