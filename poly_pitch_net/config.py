import poly_pitch_net as ppn
###############################################################################
# Audio parameters FretNet + OriginalCrep
###############################################################################


# Width of a pitch bin
CENTS_PER_BIN = 5.  # cents

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
PITCH_BINS = 1440

# Audio sample rate
SAMPLE_RATE = 22050 # hz

# Size of the analysis window
WINDOW_SIZE = 1024  # samples

# Number of consecutive frames within each example fed to the model
NUM_FRAMES = 200

NO_STRINGS = 6

DROPOUT = None

# Whether to peak-normalize CREPE input audio
NORMALIZE_INPUT = True

# Type of model normalization
NORMALIZATION = 'batch'

###############################################################################
# LOSS TYPES
###############################################################################

LOSS_ONE_HOT = 'one-hot'
LOSS_GAUSS = 'gauss-blur'

###############################################################################
# GuitarSet Dataset parameters
###############################################################################

GSET_BASE_DIR = ppn.tools.misc.get_project_root().parent.parent / 'Datasets' / 'GuitarSet'
GSET_CACHE = ppn.tools.misc.get_project_root().parent.parent / 'generated' / 'data'
GSET_CACHE_TRAIN = GSET_CACHE / 'train'
GSET_CACHE_PYTEST = GSET_CACHE / 'pytest'
GSET_CACHE_VAL = GSET_CACHE / 'val'

# guitarset recording charachteristics
GSET_SAMPLE_RATE = 44100
GSET_HOP_LEN = 256
GSET_TIME_STEP = GSET_HOP_LEN / GSET_SAMPLE_RATE
GSET_PLAYERS = 6

# HCQT data processor paramters
HCQT_DIM_IN = 144
HCQT_NO_HARMONICS = 6


##########################################
# MODEL and DATASET dictionary keys
##########################################

# model only 
KEY_PITCH_WG_AVG = 'pitch_wg_average'
KEY_PITCH_CENTERS = 'pitch_centers'
KEY_PITCH_LOGITS = 'pitch_logits'
KEY_PITCH_ARRAY_CENTS = 'pitch_array_cents'
KEY_PITCH_ARRAY_HZ = 'pitch_array_hz'

# guitarset
KEY_FS = 'fs'
KEY_FEATURES = 'features'
KEY_AUDIO = 'audio'
KEY_AUDIO_CHUNKS = 'audio_chunks'
KEY_TIMES = 'times'
KEY_TRACK = 'track'
KEY_MULTIPITCH = 'multi_pitch'
KEY_PITCH_ARRAY = 'pitch_array'
KEY_FEATURES = 'features'

# for convert.py
NO_PITCH_BIN = -1000000

###############################################################################
# Training parameters
###############################################################################

# Batch size
BATCH_SIZE = 30

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

LOSS_BCE = 'binary_cross_entropy_with_logits'

# Number of training steps
STEPS = 2500

# Number of frames used during training
NUM_TRAINING_FRAMES = 1

# Number of data loading worker threads
NUM_WORKERS = 4

# Seed for all random number generators
RANDOM_SEED = 1234

# Whether to only use voiced start frames
VOICED_ONLY = False

