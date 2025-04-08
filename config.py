# Political boundaries
ADMIN_0_PATH = 'data/geocells/admin_data/adm_0.gpkg'
ADMIN_1_PATH = 'data/geocells/admin_data/adm_1.gpkg'
ADMIN_2_PATH = 'data/geocells/admin_data/adm_2.gpkg'

# Other paths 
FUSED_GEOCELL_PATH = "data/geocells/cells/inat2017_fused_cells.npy"
GEOCELL_PATH = 'data/geocells/cells/geocells.csv' 
LOC_PATH = 'data/training/locations/locations.csv'

CRS = 'EPSG:4087'

# Geocell creation
MIN_CELL_SIZE = 100
MAX_CELL_SIZE = 500 

# Haversine smoothing constant
LABEL_SMOOTHING_CONSTANT = 65 
LABEL_SMOOTHING_MONTHS = 0.3