import kagglehub

# Download latest version
path = kagglehub.model_download("google/bird-vocalization-classifier/tensorFlow2/perch_v2")

print("Path to model files:", path)