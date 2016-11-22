# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Enhancement Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#

import numpy as np
import tensorflow as tf
import os
import time


def get_vars_main_target(t_vars, main_name='main_q', target_name='target_q'):
    main_vars = [var for var in t_vars if var.name.startswith(main_name)]
    target_vars = [var for var in t_vars if var.name.startswith(target_name)]
    for x in main_vars:
        assert x not in target_vars
    for x in target_vars:
        assert x not in main_vars
    for x in t_vars:
        assert x in main_vars or x in target_vars

    return {'main_vars': main_vars, 'target_vars': target_vars}


def processState(state1):
    return np.reshape(state1, [21168])


def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS) / duration * t)]
        except:
            x = salIMGS[-1]
        return x

    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience:
        mask = mpy.VideoClip(make_mask, ismask=True, duration=duration)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps=len(images) / duration, verbose=False)
        # clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps=len(images) / duration, verbose=False)


def save_model(saver, sess, model_dir, model_name, step=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saver.save(sess, model_dir + 'weights_', global_step=step)


def load_model(saver, sess, model_dir):
    print(" [*] Loading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        fname = os.path.join(model_dir, ckpt_name)
        saver.restore(sess, fname)
        print(" [*] Load SUCCESS: %s" % fname)
        return True
    else:
        print(" [!] Load FAILED: %s" % model_dir)
    return False


def get_time():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())


def get_input_shape(x):
    "Return input shape"
    if isinstance(x, tf.Tensor):
        return x.get_shape().as_list()
    elif type(x) in [np.ndarray, list, tuple]:
        return np.shape(x)
    else:
        raise Exception("Invalid input layer")
