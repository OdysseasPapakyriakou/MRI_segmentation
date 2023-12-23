# Project description
The project builds, trains, and tests a U-net model for 
segmenting white matter hyperintensity (through brain MRI scans) 
in subjects from three different institutions. The goal is to 
explore multi-institute training and how it affects the 
model's performance.

Therefore, the model is trained eight times on data from 
every combination from one to three institutions and tested 
on data again from every combination from one to three institutions.
The three institutions are:
1. Utrecht Medical Center (UMC)
2. Vrije University Medica Center (VUmc) in Amsterdam
3. National University Health System (NUHS) in Singapore

Results showed that training the U-net model on a combination 
of scans from UMC and NUHS together, produces segmenation masks
for white matter hyperintensity the yield the greatest similarity
to the groundtruth masks across all institutions.

This means, for example, that training on data from UMC and NUHS 
produces segmentation masks for the VUmc data that are more accurate
compared to segmentations masks when trained on the VUmc data,
or any other institute combination.

This suggests that the U-net model trained on the UMC and NUHS scans
is the most generalizable, and that multi-institute training has
a notable effect on the model's performance.

# Organization of the code
All source code is in the code folder. It is run in 
*main_train_test.ipunb* in Google Colab, where the data 
are also loaded from Google Drive. 

The data are loaded using PyTorch's dataloaders, which handles 
batching and train and validation split. The custom classes that 
inherit PyTorch's Dataset and Datamodule classes are in 
*dataset_dataloading.py*. 

The dice loss that is used for training the U-net model is defined in
*helper_functions.py*, and the model itself is defined in
*unet_model.py*.

Zero padding has been implemented in *zero_padding.ipynb*
to bring the scans into the same dimensions. This has already been
run and the training and test data used now have zero padding.

# Report
The results and more detail about the project can be found in *report.pdf*

