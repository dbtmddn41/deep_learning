import tensorflow as tf

class config:
    batch_size = 32

cfg = config()

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def create_example(data, target):
    feature = {
        "audio": float_feature_list(data),
        "target": int64_feature(target),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def save_tfrecord(data, target, idx):
    with tf.io.TFRecordWriter(f"train_data_{idx}.tfrec") as writer:
        for audio, label in zip(data, target):
            example = create_example(audio, label)
            writer.write(example.SerializeToString())

def prepare_tfrecord(wave_files, labels):
    datas = []
    targets = []
    for i, (wave_file, label) in enumerate(zip(wave_files, labels)):
        file = tf.io.read_file(wave_file)
        # 2. Decode the wav file
        audio, sr = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        datas.append(audio)
        targets.append(label)
    save_tfrecord(datas, targets, 0)

def parse_tfrecord_fn(example):
    feature_description = {
        "audio": tf.io.VarLenFeature(tf.float32),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["audio"] = tf.sparse.to_dense(example["audio"])
    return example

def get_dataset():
    AUTOTUNE = tf.data.AUTOTUNE
    filenames = tf.io.gfile.glob(f"*.tfrec")
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        # .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(cfg.batch_size * 10)
        .batch(cfg.batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset