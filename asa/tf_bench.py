#!/usr/bin/env python3
import tensorflow as tf, time
print("Devices:", tf.config.list_physical_devices('GPU'))
n = 2048
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    A = tf.random.normal((n,n), dtype=tf.float32)
    B = tf.random.normal((n,n), dtype=tf.float32)
    C = tf.linalg.matmul(A,B)  # warmup
    t0 = time.time()
    C = tf.linalg.matmul(A,B)
    print("Time:", time.time()-t0, "s")
