# Generative DEM Void Filling With Terrain Feature-Guided Transfer Learning Assisted by Remote Sensing Images
It is the pytorch implementation of the letter: Generative DEM Void Filling With Terrain Feature-Guided Transfer Learning Assisted by Remote Sensing Images. Here is the first commit, more illustrations of the implementation will be released soon.
# Our environments
python==3.8.11
GDAL==2.4.1
scikit_image==0.18.1
matplotlib==3.4.2
numpy==1.22.4
kornia==0.6.1
torch==1.9.1
torchvision==0.10.0
# Dataset
* Description: This dataset included a part of 3× test sets, which were obtained from TanDEM-X and Sentienl-2. The dataset contains a total of 100 256*256 data pairs. Each data pair contains one complete DEM, one Mask and one RS image. Complete DEM, remote sensing images can be obtained from https://www.intelligence-airbusds.com and https://earthexplorer.usgs.gov/, respectively.  
* Download the dataset from the following link: [Baidu cloud disk](https://pan.baidu.com/s/190lAUq4Ol5fdJXfo-m9CsA) (code:cx7q)
# Acknowledgement
Thanks for the excellent work and codebase of GateConv! 
