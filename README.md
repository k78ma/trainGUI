# Training GUI Documentation
**Important:** Some files are missing (too large to upload to GitHub)

## Requirements:
- Python 3.7
- PyTorch 1.8.0
- XGBoost 1.6.2
- Transformers 4.23.1
<br>
For a full list of requirements, see [requirements.txt](https://github.com/k78ma/trainGUI/blob/main/requirements.txt)

## Starting the GUI through terminal
1. Create a conda environment named `gui` with the required packages above.
2. Activate the environment using `conda gui`.
3. Switch to the correct directory with `cd trainGUI`.
4. Run the GUI using `python gui.py`.

## Usage of GUI
- Set session name, training data, and output directory on the "Main" tab.
- Help and quit buttons are available at the bottom. <br> <br>
![Main Page](https://user-images.githubusercontent.com/77073162/208039476-fa9da6cc-8260-43f8-af8f-236fdea8ab2b.png)
---
- Then, training hyperparameters can be set using the "Parameters" tab.
- Recommended parameters can be automatically inserted (shown in picture) using the bottom-left "Use default" button.
- After inputting parameters, they can be set and saved using the "Set parameters" button. <br> <br>
![Parameters](https://user-images.githubusercontent.com/77073162/208040311-0b4a4437-de2f-461c-b39d-bb8d4d687fb2.png)
---
- After setting parameters, use the "Start training" button to begin training.
- The top progress bar will show the total progress of the training process (current epoch / total epochs).
- The bottom progress bar will show the progress of the current epoch.
- After finishing, a pop-up will appear indicating that training has finished. <br> <br>
![Progress Bar](https://user-images.githubusercontent.com/77073162/208041536-9b932eb4-a284-46f8-b933-7867b94db08a.png) &nbsp;	&nbsp;	&nbsp;	![Pop-up](https://user-images.githubusercontent.com/77073162/208041943-b6cdd719-e53d-412f-acca-b37f1ec4eb3e.png)
---
- After training, loss, FP rate, and FN rate plots are available in their respective tabs. An example loss graph is shown below. 
- The plots support panning, zooming, saving, and custom configuration, all done through the bottom toolbar. <br> <br>
![Loss plot](https://user-images.githubusercontent.com/77073162/208042230-7a5ac3b1-d493-43e6-aff6-9539bcea29b6.png)

