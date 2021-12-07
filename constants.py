N_SCALES = 3  # [1.0, 0.9, 1.1]
N_FEATURES = 2  # [map3, map4]
N_CHANNELS = N_SCALES * N_FEATURES

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# Directories
LOGS_DIR = './logs'
DATA_DIR = './data'
GENERATED_DIR = './generated'

ANNOTATION_DIR = f'{DATA_DIR}/annotation_FSC147_384.json'
SPLIT_DIR = f'{DATA_DIR}/Train_Test_Val_FSC_147.json'

ORIGINAL_IMAGES_DIR = f'{DATA_DIR}/images_384_VarV2'
ORIGINAL_DENSITIES_DIR = f'{DATA_DIR}/gt_density_map_adaptive_384_VarV2'

RESIZED_IMAGES_DIR = f'{GENERATED_DIR}/resized_images'
RESIZED_DENSITIES_DIR = f'{GENERATED_DIR}/resized_densities'

PREPROCESSED_IMAGE_FEATURES_DIR = f'{GENERATED_DIR}/preprocessed_image_features'
PREPROCESSED_DENSITIES_DIR = f'{GENERATED_DIR}/preprocessed_densities'

POINTS_COUNT_FILE = f'{GENERATED_DIR}/points_count.json'
IMAGE_COORDS_FILE = f'{GENERATED_DIR}/image_coords.json'
BBOXES_COORDS_FILE = f'{GENERATED_DIR}/bboxes_coords.json'
CORRELATION_FILE = f'{GENERATED_DIR}/correlation.json'
