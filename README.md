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
2. NUMPY_ARCHIVE_PATH=/Users/ronenrayten/Downloads/ #the root directory of the processed numpy samples. This is the root directory where you are going to install the converted repository. The path is being used by the data loader to load the bach of files per batch size.

If you want to recreate the numpy repository from scratch:
1. unzip the original archive downloaded from Kaggle to the root_dir as defined above.
2. set the numpy_archive_path to the directory to where the converted images are going to be saved.
3. set CREATE_REPOSITORY in .env to true andrRun the cell where  #create_repository and generate images comment


```
If you have the converted repository already, just set the above var paths correctly and you are set and set CREATE_REPOSITORY in env to false.

The code is creating a dataset, mini batch is set to 10 (variable that can be changed), and the loader.
Transformer - normalize the batch (mean and std) and convert to tensor
The Tensors are (channels x W x H x Voxels) which is for the image (4x130x160x128) and the label (4x130x160x128). The label has 4 classes and transformed to categorical, this is why the first dim is 4
