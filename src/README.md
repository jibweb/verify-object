Scripts:
- `main.py`: Runs the optimization on all images of the dataset. Outputs estimated object poses.
- `evaluation.py`: Computes and plots the error of the estimated poses.

Packages:
- `pose/`: The inverse rendering model and optimization loop.
- `mask/`: Everything related to the mask-based loss.
- `collision/`: Everything related to the collision-based loss.
- `contour/`: Everything related to the contour-based loss.
- `utility/`: Helper classes and functions.