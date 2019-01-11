## Request
 - python >= 3.6.0
 - pytorch >= 0.4.0
 - numpy >= 1.14
 - cuda >= 9.0
 - torchvision >= 0.2.1
 
## Pretrained Model
 We put our trained model in the path ./data/ and you can use them directly. To repeat our work with the code, please follow these steps:
 
 1. Train our AlexNet with Cifar-10 dataset. Please put the dataset directory "cifar-10-batches-py" into the "data" directory. Then run the command:
*python cifar_alex.py*
 3. Train the reconstuct net. If you want to train the reconstuct_net, please make sure the features are extracted from the fourth hidden layer of the AlexNet. And if you want to train the reconstruct_net2, please make sure the features are extracted from the fifth hidden layer. We are sorry for the inconvenient because you have to adjust this in the code of "cifar_alex.py". Then run the command:
*python reconstruct.py* or
*python reconstruct_v2.py*
These two nets will serve as a control group in out project.
 
## Compress Model
 To apply the ADMM algorithm to the reconstruct net, please run the command:
 *python reconstruct_pruned_new.py* or
 *python reconstructv2_pruned_new.py* or
 *python reconstructv2_pruned.py* (this one partly compress the model)
 In the end, you can run *python showResult.py* to visualize the inverted images.
