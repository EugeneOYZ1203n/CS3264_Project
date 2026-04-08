# 0. Setup the environment.
conda create --name vq python=3.11
conda activate vq
pip install ".[dev]"

DATA_DIR=/output/extract_poses/
POSES_DIR=/output/poses/sgsl

# 1. Downloads lots of poses from the bucket. (about 508GB)
sbatch scripts/sync_bucket.sh "$POSES_DIR"
# Check the number of files (should be above 500k)
find "$POSES_DIR" -type f -name "*.pose" | wc -l

# 2. Collect normalization data
sbatch scripts/extract_mean_std.sh "$POSES_DIR"

# 3. Creates a ZIP file of the poses after normalizing them. (about 45GB)
sbatch scripts/zip_dataset.sh "$POSES_DIR" "$DATA_DIR/normalized.zip"

# 4. Trains the model and reports to `wandb`.
sbatch scripts/train_model.sh "$DATA_DIR/normalized.zip"