
# Python STD
from enum import auto, Enum, unique
import math

# Data
import numpy as np
# import pandas as pd
import scipy as sp

# Machine Learning

# Deep Learning
import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Image Processing / Computer Vision

# Optimization

# Auxiliary

# Visualization
import matplotlib.pyplot as plt

# Miscellaneous
import time

# Course Packages
from DeepLearningBlocks import NNMode


# Typing
from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union

# Auxiliary Classes

class TBLogger():
    def __init__( self, logDir: Optional[str] = None ) -> None:

        self.oTBWriter  = SummaryWriter(log_dir = logDir)
        self.iiEpcoh    = 0
        self.iiItr      = 0
        
        pass

    def close( self ) -> None:

        self.oTBWriter.close()


# Auxiliary Functions

def ResetModelWeights( oModel: nn.Module ) -> None:
    # https://discuss.pytorch.org/t/19180

    for layer in oModel.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        elif hasattr(layer, 'children'):
            for child in layer.children():
                ResetModelWeights(child)

# Initialization Function

def InitWeights( oLayer: nn.Module ) -> None:
    # Use: oModel.apply(InitWeights) #<! Applies the function on all layers
    if isinstance(oLayer, nn.Linear):
        nn.init.kaiming_normal_(oLayer.weight.data)

def InitWeightsKaiNorm( oLayer: nn.Module, tuLyrClas: Tuple = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Bilinear, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d) ) -> None:
    # Use: oModel.apply(InitWeights) #<! Applies the function on all layers
    if isinstance(oLayer, tuLyrClas):
        nn.init.kaiming_normal_(oLayer.weight.data)


def RunEpoch(oModel: nn.Module, dlData: DataLoader, hL: Callable, hS: Callable, oOpt: Optional[Optimizer] = None, opMode: NNMode = NNMode.TRAIN) -> Tuple[float, float]:
    """
    Runs a single Epoch (Train / Test) of a model.  
    Input:
        oModel      - PyTorch `nn.Module` object.
        dlData      - PyTorch `Dataloader` object.
        hL          - Callable for the Loss function.
        hS          - Callable for the Score function.
        oOpt        - PyTorch `Optimizer` object.
        opMode      - An `NNMode` to set the mode of operation.
    Output:
        valLoss     - Scalar of the loss.
        valScore    - Scalar of the score.
    Remarks:
      - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
      - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
      - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a scalar `valScore` of the score.
      - The optimizer is required for training mode.
    """
    
    epochLoss = 0.0
    epochScore = 0.0
    numSamples = 0
    numBatches = len(dlData)
    start = time.time()
    runDevice = next(oModel.parameters()).device  # Get the device (CPU/GPU) where the model is located

    if opMode == NNMode.TRAIN:
        oModel.train(True)
    elif opMode == NNMode.INFERENCE:
        oModel.eval()
    else:
        raise ValueError(f'The `opMode` value {opMode} is not supported!')

    for ii, (mX, vY) in enumerate(dlData):
        start_epcoh = time.time()
        mX = mX.to(runDevice)
        vY = vY.to(runDevice)

        batchSize = mX.shape[0]

        if opMode == NNMode.TRAIN:
            mZ = oModel(mX)
            valLoss = hL(mZ, vY)

            oOpt.zero_grad()
            valLoss.backward()
            oOpt.step()
        else:
            with torch.no_grad():
                mZ = oModel(mX)
                valLoss = hL(mZ, vY)

        with torch.no_grad():
            # Convert vY and mZ to index tensors if they are not already
            vY_index = vY.argmax(dim=1) if vY.dtype == torch.float64 else vY
            mZ_index = mZ.argmax(dim=1) if mZ.dtype == torch.float32 else mZ

            #print(f"Shape of mZ_index: {mZ_index.shape}, dtype: {mZ_index.dtype}")
            #print(f"Shape of vY_index: {vY_index.shape}, dtype: {vY_index.dtype}")
            
            try:
                valScore = hS(mZ_index, vY_index)
                #print(f"Shape of preds: {mZ_index.shape}, dtype: {mZ_index.dtype}")
                #print(f"Shape of target: {vY_index.shape}, dtype: {vY_index.dtype}")
            except Exception as e:
                print(f"Error during scoring: {e}")
                raise e

            epochLoss += batchSize * valLoss.item()
            epochScore += batchSize * valScore.item()
            numSamples += batchSize
            epochTime = time.time() - start_epcoh

        print(f'{"Train" if opMode == NNMode.TRAIN else "Val"} - Iteration: {(ii + 1):3d} / {numBatches}: loss = {valLoss:.6f} epoch time:{epochTime:5.2f}')
    total_time = time.time() - start
    print('', end='\r')
    print(f'{"Train" if opMode == NNMode.TRAIN else "Val"} - Total Time:{total_time:5.2f} |', end='')

    return epochLoss / numSamples, epochScore / numSamples


# Training Model Loop Function

def TrainModel(oModel: nn.Module, dlTrain: DataLoader, dlVal: DataLoader, oOpt: Optimizer, numEpoch: int, hL: Callable, hS: Callable, oSch: Optional[LRScheduler] = None, oTBWriter: Optional[SummaryWriter] = None) -> Tuple[nn.Module, List, List, List, List]:
    lTrainLoss = []
    lTrainScore = []
    lValLoss = []
    lValScore = []
    lLearnRate = []

    bestScore = -1e9

    learnRate = oOpt.param_groups[0]['lr']

    for ii in range(numEpoch):
        startTime = time.time()
        trainLoss, trainScr = RunEpoch(oModel, dlTrain, hL, hS, oOpt, opMode = NNMode.TRAIN)
        valLoss, valScr = RunEpoch(oModel, dlVal, hL, hS, oOpt, opMode = NNMode.INFERENCE)
        if oSch is not None:
            learnRate = oSch.get_last_lr()[0]
            oSch.step()
        epochTime = time.time() - startTime

        lTrainLoss.append(trainLoss)
        lTrainScore.append(trainScr)
        lValLoss.append(valLoss)
        lValScore.append(valScr)
        lLearnRate.append(learnRate)

        if oTBWriter is not None:
            oTBWriter.add_scalars('Loss (Epoch)', {'Train': trainLoss, 'Validation': valLoss}, ii)
            oTBWriter.add_scalars('Score (Epoch)', {'Train': trainScr, 'Validation': valScr}, ii)
            oTBWriter.add_scalar('Learning Rate', learnRate, ii)

        print('Epoch ' f'{(ii + 1):4d} / ' f'{numEpoch}:', end='')
        print(' | Train Loss: ' f'{trainLoss:6.3f}', end='')
        print(' | Val Loss: ' f'{valLoss:6.3f}', end='')
        print(' | Train Score: ' f'{trainScr:6.3f}', end='')
        print(' | Val Score: ' f'{valScr:6.3f}', end='')
        print(' | Epoch Time: ' f'{epochTime:5.2f}', end='')

        if valScr > bestScore:
            bestScore = valScr
            try:
                dCheckPoint = {'Model': oModel.state_dict(), 'Optimizer': oOpt.state_dict()}
                if oSch is not None:
                    dCheckPoint['Scheduler'] = oSch.state_dict()
                torch.save(dCheckPoint, 'BestModel.pt')
                print(' | <-- Checkpoint!', end='')
            except:
                print(' | <-- Failed!', end='')
        print(' |')

    # Load best model ("Early Stopping")
    # dCheckPoint = torch.load('BestModel.pt')
    # oModel.load_state_dict(dCheckPoint['Model'])

    return oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate



# Auxiliary Blocks

# Residual Class
# Residual Block:
# - https://scribe.rip/471810e894ed.
# - https://stackoverflow.com/questions/57229054.
# - https://wandb.ai/amanarora/Written-Reports/reports/Understanding-ResNets-A-Deep-Dive-into-Residual-Networks-with-PyTorch--Vmlldzo1MDAxMTk5
# For nn. vs. F. see:
# - https://discuss.pytorch.org/t/31857
# - https://stackoverflow.com/questions/53419474

# Simple Residual Block
class ResidualBlock( nn.Module ):
    def __init__( self, numChnl: int ) -> None:
        super(ResidualBlock, self).__init__()
        
        self.oConv2D1       = nn.Conv2d(numChnl, numChnl, kernel_size = 3, padding = 1, bias = False)
        self.oBatchNorm1    = nn.BatchNorm2d(numChnl)
        self.oReLU1         = nn.ReLU(inplace = True)
        self.oConv2D2       = nn.Conv2d(numChnl, numChnl, kernel_size = 3, padding = 1, bias = False)
        self.oBatchNorm2    = nn.BatchNorm2d(numChnl)
        self.oReLU2         = nn.ReLU(inplace = True) #<! No need for it, for better visualization
            
    def forward( self: Self, tX: torch.Tensor ) -> torch.Tensor:
        
        tY = self.oReLU(self.oBatchNorm1(self.oConv2D1(tX)))
        tY = self.oBatchNorm2(self.oConv2D2(tY))
        tY += tX
        tY = self.oReLU(tY)
		
        return tY