# CISC881FinalProject

## 3D U-Net training

The 3DUnetCISC881.ipynb notebook contains the training logs and evaluation of all loss function models 
that were trained. 

The FocalLossModelPrediction.npy 3D Numpy array is the catheter predictions of the focal loss model that 
was predicted from one of the testing TRUS images.

## Curved Hough transform

The other python files are for the implementation of the iterative curved Hough transform. In order to run and 
visualize the Hough transform on the FocalLossModelPrediction.npy array, run the following line in your console:

<code>
python CatheterPrediction.py
</code>


Three 3D plots will be generated: The catheter predictions from the 3D U-Net model, the catheter predictions 
and the curved catheters, and just the curved catheters.