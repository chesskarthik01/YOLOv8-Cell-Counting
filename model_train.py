# %%
from ultralytics import YOLO
import seaborn as sns
sns.set_style('darkgrid')
# %%
# Load model
model = YOLO('yolov8n.pt')
# %%
# Train the model
results = model.train(task='detect', mode='train', model=model,
                      data='config.yaml', imgsz=1024, epochs=300)

# %%
