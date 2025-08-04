
from sklearn.pipeline import Pipeline
from features.image_loader import ImageLoader

def create_image_pipeline(image_dir, image_size=(64, 64)):
    return Pipeline([
        ("loader", ImageLoader(image_dir=image_dir, image_size=image_size))
    ])