import os
import os.path as osp

class Config:
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    label_dir = osp.join(root_dir, 'label')
    output_dir = osp.join(root_dir, 'output')
    
    dataFile = 'MIMM5021-IR-local-annotation-2d-3d-gray.json'
    labelFile = "Mara_4_18_19_REVIEWED_5813d4880fbb2c2af0077870_5813d4ad0fbb2c2af0077871.json"
    
    dataPath = osp.join(data_dir,dataFile)
    labelPath = osp.join(label_dir, labelFile)

    neuronNum = 128
    epochs = 20000
    ptsNum = 10 # range 0 ~ 14 keypoints
    batchSize = 1000
    logInterval = 1000
    testPortion = 0.2
    trainPortion = 0.7
    
