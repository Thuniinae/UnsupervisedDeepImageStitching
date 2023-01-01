#WORKSPACE = '../../ImageAlignment/output/'
WORKSPACE = '../../../Multi-Grid-Deep-Homography/output/'
#training dataset path
TRAIN_FOLDER = WORKSPACE+'training'

#testing dataset path
TEST_FOLDER = WORKSPACE+'testing'

#GPU index
GPU = '0'

#batch size for training
TRAIN_BATCH_SIZE = 1

#batch size for testing
TEST_BATCH_SIZE = 1

#num of iters
ITERATIONS = 200000

# checkpoints path
SNAPSHOT_DIR = "./checkpoint_pretrained"

#sumary path
SUMMARY_DIR = "./summary"
