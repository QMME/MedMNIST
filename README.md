# MedMNIST 

  This is the project for EE228.

  Put the data in folder `input`.
  
  For instance, to run PathMNIST
  
      python train.py --data_name pathmnist --input_root <path/to/input/folder> --output_root <path/to/output/folder> --epoch 100 
  
  To test the `.pth` file, run

      python test.py --data_name pathmnist --input_root <path/to/input/folder> --load_root <path/to/output/folder>  

  The pretrained models using CutMix are in `cutmix_output_18` and `cutmix_output_50`. Pretrained models of origins are in `Resnet18-output` and `Resnet50-output`.


