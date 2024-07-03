Cofig env variables:
VS Code Settings:

Add a setting in your settings.json file (accessible via the Command Palette Ctrl+Shift+P -> "Preferences: Open Settings (JSON)"):
Put your full path to .env
```
{
  "python.envFile": "${workspaceFolder}/.env"
}
```
Copy .env.example as .env and change ROOT_DIR and NUMPY_ARCHIVE_PATH


Update the following variables:
1. ROOT_DIR=/Users/ronenrayten/Downloads/archive2021/ #The directeroy of the original Kaggel archive. This is where you unzipped the original gz file
2. NUMPY_ARCHIVE_PATH=/Users/ronenrayten/Downloads/ #the root directory of the processed numpy.gz samples. This is the root directory where you are going to install the converted repository. The path is being used by the data loader to load the bach of files per batch size.

number_of_channels is claculated by summing the number of images were enabled as TRUE samples in the .env file. For example:
ENABLE_T1=false
ENABLE_T1CE=true
ENABLE_T2=true
ENABLE_FLAIR=true

than the number of channels is set to 3.

*NOTE: the mean and std are set as const and were calculated for 4 channels. The array is adjucted automatically according to the channels selected.


If you want to recreate the numpy repository from scratch:
1. unzip the original archive downloaded from Kaggle to the root_dir as defined above.
2. set the numpy_archive_path to the directory to where the converted images are going to be saved.
3. set CREATE_REPOSITORY in .env to true andrRun the cell where  #create_repository and generate images comment



If you have the converted repository already, just set the above var paths correctly and you are set and set CREATE_REPOSITORY in env to false.

The code is creating a dataset, mini batch is set to 1 (variable that can be changed), and the loader.
Transformer - normalize the batch (mean and std) and convert to tensor
The Tensors are (channels x W x H x Voxels) which is for the image (num_of_channelsx130x160x128) and the label (4x130x160x128). The label has 4 classes and transformed to categorical, this is why the first dim is 4


# Adding loss module:
1. pip install kornia
2. Add loss.py (download loss.py from our gitHub) to your brain_main_2021_new.ipynb:
   from loss import FocalLoss
3. After cthe cell where we alculate the mean and std add the following cell:
`#compute class weights for the loss function
class_weights = sum(label_count) / (4 * np.array(label_count))
# Convert class weights to a list or tensor for use in FocalLoss
class_weights = torch.tensor(class_weights, dtype=torch.float32)`
4. Change the loss function as follows:
`hL = FocalLoss(alpha=0.5,gamma=0.7,weight=class_weights,reduction='mean')`

