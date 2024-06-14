Update the following variables:
1. root_dir = '/Users/ronenrayten/Downloads/archive2021/' #The directeroy of the original Kaggel archive. This is where you unzipped the original gz file
2. numpy_archive_path = '/Users/ronenrayten/Downloads/' #the root directory of the processed numpy samples. This is the root directory where you are going to install the converted repository. The path is being used by the data loader to load the bach of files per batch size.

If you want to recreate the numpy repository from scratch:
1. unzip the original archive downloaded from Kaggle to the root_dir as defined above.
2. set the numpy_archive_path to the directory to where the converted images are going to be saved.
3. Run the cell (currently it is marked out in the notebook):
```python
#create_repository and generate images
 files = get_file_lists()
 generate_images(files)

```
If you have the converted repository already, just set the above var paths correctly and you are set.

The code is creating a dataset, mini batch is set to 10 (variable that can be changed), and the loader.
Transformer - normalize the batch (mean and std) and convert to tensor
The Tensors are (channels x W x H x Voxels) which is for the image (4x130x160x128) and the label (1x130x160x128)
