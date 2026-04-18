# Archive

## azure_plan/
Original Azure ML training plan. Archived because the Azure Education subscription
did not allow GPU compute quota, so training was migrated to Google Colab Pro.

The WGAN-GP trainer (trainer.py) was also designed for Azure CUDA and could not
run on the local Apple M5 Mac (MPS does not support create_graph=True in autograd).

## 2d_model_drafts/
Early 2D CNN prototypes (Conv2d). Superseded by 3D models in baseline.py.
Not used in any final pipeline.
