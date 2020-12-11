# Author: Senthil Palanisamy
# This file contains a class for running deeplabcut inference on a single frame

import sys
import os.path
# sys.path.append('/home/senthil/anaconda3/envs/DLC-GPU/lib/python3.7/site-packages')
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
import time
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from deeplabcut.utils import auxiliaryfunctions
import cv2
from skimage.util import img_as_ubyte


from deeplabcut.utils.auxfun_videos import imread

class DLC_frame_inference:
  def __init__(self, config ,shuffle=1, trainingsetindex=0, 
                     gputouse=None, rgb=True):
    '''
    The constructor loads the deeplearning model from the specified path
    config - path where the model config.yaml file is stored
    shuffle - Not sure how it affects the algorithm. It was set to one
             inside the core library. So keep sticking to 1 unless you 
             have a strong need to change
    trainingsetindex - always set it 0. Not sure of its effect. Used deep inside
                      the core library
    gputouse - gpu card number to use. The algorithm uses GPU by default 
               when tensorflow GPU version is installed. If GPU is not
               getting utlised, this is a good parameter to check. But setting
               this to None in our present system still utilises the GPU.
    rgd - set to true if we use a RGB image. For grayscale image, which is our
          case this is always False
    '''

    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training

    if gputouse is not None:  # gpu selection
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

    vers = (tf.__version__).split('.')

    if int(vers[0]) == 1 and int(vers[1]) > 12:
        TF = tf.compat.v1
    else:
        TF = tf

    TF.reset_default_graph()

    cfg = auxiliaryfunctions.read_config(config)
    cfg['batch_size'] = 1
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    modelfolder=os.path.join(cfg["project_path"],
                             str(auxiliaryfunctions.GetModelFolder
                                      (trainFraction,shuffle,cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError("It seems the model for shuffle %s and" 
                                "trainFraction %s does not exist."%(shuffle,
                                                             trainFraction))
    # Check which snapshots are available and sort them by # iterations
    try:
      Snapshots = np.array([fn.split('.')[0]for fn in 
                           os.listdir(os.path.join(modelfolder , 
                                                   'train'))if "index" in fn])
    except FileNotFoundError:
        raise FileNotFoundError("Snapshots not found! It seems the dataset"
                                "for shuffle %s has not been trained/does not"
                                "exist.\n Please train it before using it to"
                                "analyze videos.\n Use the "
                                "function 'train_network' to train the "
                                "network for shuffle %s."%(shuffle,shuffle))

    if cfg['snapshotindex'] == 'all':
        print("Snapshotindex is set to 'all' in the config.yaml file. "
              "Running video analysis with all snapshots is very costly!"
              "Use the function 'evaluate_network' to choose the best" 
              "the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex=cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]

    #update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size'] = cfg['batch_size']

    # Name for scorer:

    # update number of outputs and adjust pandas indices
    dlc_cfg['num_outputs'] = cfg.get('num_outputs', 1)

    # Name for scorer:
    self.sess, self.inputs, self.outputs = predict.setup_GPUpose_prediction(dlc_cfg)

    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
    self.rgb = rgb
    self.cfg = cfg
    self.dlc_cfg = dlc_cfg
    self.pose_tensor = predict.extract_GPUprediction(self.outputs, self.dlc_cfg)

    if self.cfg['cropping']:
      self.ny, self.nx=checkcropping(self.cfg,cap)


  def infer_mice_pose(self, frame):
    '''
    A function for predicitng the mice pose for a given frame
    frame - image for which mice pose prediction is to be made
    Returns:-
    A N X 2 array of points where N denotes the number of mice body parts. Thus
    x,y location of each body part is given out by the network
    '''

    PredictedData = np.zeros((1, 3 * len(self.dlc_cfg['all_joints_names'])))
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if self.cfg['cropping']:
        frame= img_as_ubyte(frame[self.cfg['y1']:self.cfg['y2'],
                                 self.cfg['x1']:self.cfg['x2']])
    else:
        frame = img_as_ubyte(frame)
    
    start = time.time()
    pose = self.sess.run(self.pose_tensor, 
                         feed_dict= {self.inputs: np.expand_dims(frame, 
                                                       axis=0).astype(float)})
    stop = time.time()
    print('actual inference call: ', (stop-start))
    pose[:, [0,1,2]] = pose[:, [1,0,2]]
    PredictedData[0, :] = pose.flatten()  
    return PredictedData


# A global variable is needed if this file is planned to be imported into
# other python files. Importing this initialises the object where model loading
# is done so that the loaded model can be used for repeated inference.
# A key aspect of model inference - the first run might be run very slow. Never
# time the performance of your model based on the first run. Always ignore the
# first run for timing analysis

config='./models/demo-senthil-2020-08-09/config.yaml'
inference_object = DLC_frame_inference(config)

# This is the function that can be called from C++ using pybind11
# Frame is a numpy array. Mechansim to pass an image as a numpy array from
# a C++ code is shown in the example C++ code
def infer(frame):
  inference_object.infer_mice_pose(frame) 
  cv2.imwrite('sample_test.jpeg', frame)

# if __name__=='__main__':
#     infer(0)
#     infer(0)
#     infer(0)
#     infer(0)
#     infer(0)
#     infer(0)

# if __name__=='__main__':
#     # frame = cv2.imread('./test.jpeg')
#     rgb=True
#     if rgb:
#         frame =imread('./test.jpeg')
#     else:
#         frame =imread('./test.jpeg')
#     infer(frame)
#     infer(frame)
#     infer(frame)
# #    inference_object.infer_mice_pose(frame) 
# #    inference_object.infer_mice_pose(frame) 
# #    inference_object.infer_mice_pose(frame) 
