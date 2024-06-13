Update the following variables:
1. root_dir = '/Users/ronenrayten/Downloads/archive2021/' #The directeroy of the original Kaggel archive. This is where you unzipped the original gz file
2. numpy_archive_path = '/Users/ronenrayten/Downloads/' #the root directory of the processed numpy samples. This is root where you to install the converted repository. 
3. image_path = numpy_archive_path +'new_archive/images/'  # path to the processed numpy images. This is the path tho the samples, converted data (numpy files). 
4. label_path = numpy_archive_path +'new_archive/labels/'  # path to the processed numpy labeles. This is the path to the converted labels (numpy files)

If you want to recreate the numpy repository, set the paths above and run the cell (currently it is marked out):
#create_repository and generate images
 files = get_file_lists()
 generate_images(files)

If you have the converted repository already, just set the above var paths correctly and you are set.

The code is creating a dataset, mini batch is set to 10 (variable that can be changed), and the loader.
Transformer - normalize the batch (mean and std) and convert to tensor
The Tensors are (channels x W x H x Voxels) which is for the image (4x130x160x128) and the label (1x130x160x128)
