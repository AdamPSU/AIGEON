from transformers import TrainingArguments

# Political boundaries
ADMMIN_0_PATH = 'data/geocells/adm_0.gpkg'
ADMIN_1_PATH = 'data/geocells/adm_1.gpkg'
ADMIN_2_PATH = 'data/geocells/adm_2.gpkg'

# Geocell creation
MIN_CELL_SIZE = 1000 
MAX_CELL_SIZE = 2000 

# Geocells path
GEOCELL_PATH = 'data/geocells_yfcc.csv' 

# Haversine smoothing constant
LABEL_SMOOTHING_CONSTANT = 65 
LABEL_SMOOTHING_MONTHS = 0.3