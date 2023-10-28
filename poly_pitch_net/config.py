import poly_pitch_net as ppn
###############################################################################
# Audio parameters FretNet + OriginalCrep
###############################################################################


# Width of a pitch bin
CENTS_PER_BIN = 20.  # cents

# Whether to trade quantization error for noise during inference
DITHER = False

# Minimum representable frequency
FMIN = 31.  # Hz

# Distance between adjacent frames
HOPSIZE = 512 # samples

# The size of the window used for locally normal pitch decoding
LOCAL_PITCH_WINDOW_SIZE = 19

# Pitch velocity constraint for viterbi decoding
MAX_OCTAVES_PER_SECOND = 35.92

# Whether to normalize input audio to mean zero and variance one
NORMALIZE_INPUT = False

# Number of spectrogram frequency bins
NUM_FFT = 1024

# One octave in cents
OCTAVE = 1200  # cents

# Number of pitch bins to predict
PITCH_BINS = 360

# Audio sample rate
SAMPLE_RATE = 22050 # hz

# Size of the analysis window
WINDOW_SIZE = 1024  # samples

# Number of consecutive frames within each example fed to the model
NUM_FRAMES = 200

NO_STRINGS = 6


###############################################################################
# GuitarSet Dataset parameters
###############################################################################

GSET_BASE_DIR = ppn.tools.misc.get_project_root().parent.parent / 'Datasets' / 'GuitarSet'
GSET_CACHE = ppn.tools.misc.get_project_root().parent.parent / 'generated' / 'data'
GSET_CACHE_TRAIN = GSET_CACHE / 'train'
GSET_CACHE_VAL = GSET_CACHE / 'val'

###############################################################################
# Training parameters
###############################################################################


# Batch size
BATCH_SIZE = 128

# Whether to stop training when validation loss stops improving
EARLY_STOPPING = False

# Stop after this number of log intervals without validation improvements
EARLY_STOPPING_STEPS = 32

# Whether to apply Gaussian blur to binary cross-entropy loss targets
GAUSSIAN_BLUR = True

# Optimizer learning rate
LEARNING_RATE = 2e-4

# Loss function
LOSS = 'categorical_cross_entropy'

# Number of training steps
STEPS = 250000

# Number of frames used during training
NUM_TRAINING_FRAMES = 1

# Number of data loading worker threads
NUM_WORKERS = 4

# Seed for all random number generators
RANDOM_SEED = 1234

# Whether to only use voiced start frames
VOICED_ONLY = False
