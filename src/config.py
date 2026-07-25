from pathlib import Path

RANDOM_STATE = 42

# --- Chemins ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_FIGURES_DIR = ROOT_DIR / "results" / "figures"
RESULTS_TABLES_DIR = ROOT_DIR / "results" / "tables"

RAW_EXPRESSION_FILE = DATA_RAW_DIR / "TCGA_expression.csv"
RAW_LABELS_FILE = DATA_RAW_DIR / "TCGA_labels.csv"

# --- Constantes du dataset ---
UCI_DATASET_ID = 401
TUMOR_TYPES = ["BRCA", "KIRC", "COAD", "LUAD", "PRAD"]

# --- Paramètres d'analyse ---
VARIANCE_THRESHOLD = 0.5
N_PCA_COMPONENTS = 50
CV_FOLDS = 5