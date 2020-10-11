import os.path
import sys
# sys.path.append('/home/senthil/envs/malcom_lab/lib/python3.6/site-packages')
sys.path.append('/home/senthil/anaconda3/envs/DLC-GPU/lib/python3.7/site-packages')
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
import time
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from deeplabcut.utils import auxiliaryfunctions
import cv2
from skimage.util import img_as_ubyte


from deeplabcut.utils.auxfun_videos import imread




def predictPoseForFrame_GPU(cfg, dlc_cfg, sess, inputs, outputs, frame, rgb):
    ''' Non batch wise pose estimation for video cap.'''
    if cfg['cropping']:
        ny,nx=checkcropping(cfg,cap)

    pose_tensor = predict.extract_GPUprediction(outputs, dlc_cfg) #extract_output_tensor(outputs, dlc_cfg)
    nframes=1
    PredictedData = np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cfg['cropping']:
        frame= img_as_ubyte(frame[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2']])
    else:
        frame = img_as_ubyte(frame)
    
    # frame = frame.reshape((frame.shape[1], frame.shape[0], 3))
    # frame = np.expand_dims(frame, axis=2)
    # for i in range(1000):
    # pose = sess.run(pose_tensor, feed_dict={inputs: np.expand_dims(frame, axis=0).astype(float)})

    start = time.time()
    pose = sess.run(pose_tensor, feed_dict={inputs: np.expand_dims(frame, axis=0).astype(float)})
    stop = time.time()
    print('actual inference call: ', (stop-start))
    pose[:, [0,1,2]] = pose[:, [1,0,2]]
    #pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
    PredictedData[0, :] = pose.flatten()  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
    print(PredictedData)
    return PredictedData





def predictPoseForFrame(cfg, dlc_cfg, sess, inputs, outputs, im, rgb):
    ''' Batchwise prediction of pose for frame list in directory'''
    #from skimage.io import imread
    from deeplabcut.utils.auxfun_videos import imread
    print("Starting to extract posture")

    ny,nx,nc=np.shape(im)
    nframes=1
    print("Overall # of frames: ", nframes," found with (before cropping) frame dimensions: ", nx,ny)

    PredictedData = np.zeros((nframes, dlc_cfg['num_outputs'] * 3 * len(dlc_cfg['all_joints_names'])))
    batch_ind = 0 # keeps track of which image within a batch should be written to
    batch_num = 0 # keeps track of which batch you are at
    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx,ny=cfg['x2']-cfg['x1'],cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(np.shape(im)[1]) and cfg['y1']>=0 and cfg['y2']<int(np.shape(im)[0]):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')


    if cfg['cropping']:
        frame= img_as_ubyte(im[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2'],:])
    else:
        frame = img_as_ubyte(im)

    pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)

    return PredictedData,nframes,nx,ny


def predictFrame(config, frame, frametype='.png',shuffle=1,
                trainingsetindex=0,gputouse=None,save_as_csv=False,rgb=True):
    """
    Analyzed all images (of type = frametype) in a folder and stores the output in one file.

    You can crop the frames (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file.

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in the same directory, where the video is stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    directory: string
        Full path to directory containing the frames that shall be analyzed

    frametype: string, optional
        Checks for the file extension of the frames. Only images with this extension are analyzed. The default is ``.png``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    rbg: bool, optional.
        Whether to load image as rgb; Note e.g. some tiffs do not alow that option in imread, then just set this to false.

    Examples
    --------
    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1')
    --------

    If you want to analyze all frames in /analysis/project/timelapseexperiment1
    >>> deeplabcut.analyze_videos('/analysis/project/reaching-task/config.yaml','/analysis/project/timelapseexperiment1')
    --------

    Note: for test purposes one can extract all frames from a video with ffmeg, e.g. ffmpeg -i testvideo.avi thumb%04d.png
    """
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
    start_path=os.getcwd() #record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    cfg['batch_size'] = 1
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))
    # Check which snapshots are available and sort them by # iterations
    try:
      Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])
    except FileNotFoundError:
      raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))

    if cfg['snapshotindex'] == 'all':
        print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
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
    DLCscorer,DLCscorerlegacy = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations=trainingsiterations)
    if dlc_cfg['num_outputs']>1:
        if  TFGPUinference:
            print("Switching to numpy-based keypoint extraction code, as multiple point extraction is not supported by TF code currently.")
            TFGPUinference=False
        print("Extracting ", dlc_cfg['num_outputs'], "instances per bodypart")
        xyz_labs_orig = ['x', 'y', 'likelihood']
        suffix = [str(s+1) for s in range(dlc_cfg['num_outputs'])]
        suffix[0] = '' # first one has empty suffix for backwards compatibility
        xyz_labs = [x+s for s in suffix for x in xyz_labs_orig]
    else:
        xyz_labs = ['x', 'y', 'likelihood']
        TFGPUinference = True

    #sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
    if TFGPUinference:
        sess, inputs, outputs = predict.setup_GPUpose_prediction(dlc_cfg)
    else:
        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)


    pdindex = pd.MultiIndex.from_product([[DLCscorer],
                                          dlc_cfg['all_joints_names'],
                                          xyz_labs],
                                         names=['scorer', 'bodyparts', 'coords'])

    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)



    start = time.time()
    PredictedData = predictPoseForFrame_GPU(cfg,dlc_cfg, sess, inputs, outputs, frame, rgb)
    stop = time.time()
    print('inference time:  ', stop-start)

if __name__=='__main__':
   config='/home/senthil/demo-senthil-2020-08-09/config.yaml'
   # frame = cv2.imread('./test.jpeg')
   rgb=True

   if rgb:
       frame =imread('./test.jpeg')
   else:
       frame =imread('./test.jpeg')
   predictFrame(config, frame, frametype='.png',shuffle=1,
                trainingsetindex=0,gputouse=None,save_as_csv=False,rgb=True) 
