from matplotlib.colors import ListedColormap
from preprocessing import oDataTrns
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import ImageDatasetFromDisk


def visualization_model_result_as_2d(model, sample_id):
    batchSize = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mXTrain = [f"sample_{sample_id}.npy.gz"]
    vYTrain = [f"label_{sample_id}.npy.gz"]

    dsTrain = ImageDatasetFromDisk(mXTrain, vYTrain)
    dsTrain.transform = oDataTrns
    dlTrain = torch.utils.data.DataLoader(dsTrain, shuffle=True, batch_size=1 * batchSize, drop_last=True)
    tX, vY = next(iter(dlTrain))  # <! PyTorch Tensors

    # Check the shape of the tensor before passing to model
    print("Shape of input tensor:", tX.shape)
    print("Shape of random label:", vY.shape)

    # Convert random_label to single-channel by taking argmax along the channel dimension
    random_label = vY.squeeze()

    # Run the example through the model
    mode = model.to(device)
    tX = tX.to(device)
    vY = vY.to(device)
    with torch.no_grad():
        output = model(tX)  # Shape: [batch_size, num_classes, D, H, W]

    # Print the output tensor before applying argmax
    # print("Model output before argmax:", output)

    # Process the output to create a 2D visualization
    output = output.cpu()
    output = output.argmax(dim=1).squeeze()  # Get the class with the highest score
    random_label = np.argmax(random_label, axis=0)
    print('unique labels:', np.unique(random_label))
    print("Random label shape after squeeze:", random_label.shape)
    output = output.numpy()

    output_slice = output[:, :, 64]  # Adjust indexing to ensure correct slicing
    label_slice = random_label[:, :, 64]  # Ensure the label slice is correctly selected

    # Define a colormap for different classes
    cmap = ListedColormap(['black', 'red', 'green', 'blue'])

    tX = tX.cpu()
    # Visualize the result
    plt.figure(figsize=(18, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(tX[0, 0, :, :, 64], cmap='gray')  # Adjust the slice for visualization

    # Original label
    plt.subplot(1, 3, 2)
    plt.title('Original Label')
    plt.imshow(label_slice, cmap=cmap, interpolation='nearest')  # Use the discrete colormap

    # Model output
    plt.subplot(1, 3, 3)
    plt.title('Model Output')
    plt.imshow(output_slice, cmap=cmap, interpolation='nearest')  # Use the discrete colormap

    plt.show()

    print('output labels', np.unique(output_slice))
    print('output labels', np.unique(label_slice))
    print(output_slice.shape)
    print(label_slice.shape)
