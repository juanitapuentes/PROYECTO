# Panoptic Segmentation of Cell-Types Nuclei in Adenocarcinoma Colorectal Histology Images

## Set Up Environment

```sh
conda env create -f environment.yml
conda activate hovernet
pip install torch==1.6.0 torchvision==0.7.0
```
## Softlinks
In order to be able to correctly perform the following stages, you first need to generate the softlinks listed below. Make sure you place them according to how it is indicated.

| Information | Softlink |
| ------ | ------ |
| Dataset | ln -s "/media/user_home0/jpuentes/PROYECTO/dataset/CoNSeP/" "**path to dataset folder**"|
| Patches | ln -s "/media/user_home0/jpuentes/PROYECTO/dataset/training_data/ "**path to dataset folder**"|
| ResNet50 pretrained | ln -s "/media/user_home0/lvacostac/Vision/Final_Project/PROYECTO/hover_net/pretrained/ImageNet-ResNet50-Preact_pytorch.tar" "**path to pretrained folder**"|
| ResNet101 pretrained | ln -s "/media/user_home0/lvacostac/Vision/Final_Project/PROYECTO/hover_net/pretrained/nvidia_resnext101-32x4d_200821.pth.tar" "**path to pretrained folder**"|
| ResNet50 checkpoints | ln -s "/media/user_home0/jpuentes/PROYECTO/dataset/checkpoints/" "**path to dataset folder**"|
| ResNet101 checkpoints | ln -s "/media/user_home0/lvacostac/Vision/Final_Project/PROYECTO/hover_net/dataset/CHECKPOINTS/" "**path to dataset folder**"|
| ResNet50 sample tiles | ln -s "/media/user_home0/jpuentes/PROYECTO/dataset/sample_tiles/" "**path to dataset folder**"|
| ResNet101 sample tiles | ln -s "/media/user_home0/lvacostac/Vision/Final_Project/PROYECTO/hover_net/dataset/sample__tiles/" "**path to dataset folder**"|


## Training stage

This is an optional stage, given that all the checkpoints and predictions with both ResNet50 and ResNet101 pretrained weights (used for fine tunning) are already provided. However, if you want to train your own model, make sure you follow the steps below:

1. Choose whether you want to work with ResNet50 or ResNet101 as Backbone (or neither of them), and make the required adjustments to the files as indicated at the end of this README.
2. Create a new folder on the dataset folder, where you will store the checkpoints that will be generated, and another one for the corresponding predictions.
3. Go to config.py and modify the _self.log_dir_, _self.train_dir_list_ and _self.valid_dir_list_ with the paths to where the checkpoints will be saved, where the train patches are located and where the valid patches are respectively.
4. On the models/opt.py file you are able to change the main hyperparameters of the network, such as the batch size, number of epochs, weight decay and learning rate.  **_Suggestion:_** try to set the batch size to 16 or 32 max. in order to be able to execute on CUDA without running out of memory.
5. Once you have change everything you need, you'll have to run the following command on your terminal:

    ```sh
    CUDA_VISIBLE_DEVICES=gpu_number taskset -c range_of_cores python run_train.py
    ```
    Make sure you have already set up and activated the indicated environment (hovernet).

## Inference stage (Test)

To generate the predictions to the corresponding models, go to the run_tile.sh file, and update the paths as it is required. To make this file executable, first run this line on your terminal
    ```sh
    chmod +x run_tile.sh
    ```
    and then, run the file as it follows
    ```sh
    CUDA_VISIBLE_DEVICES=gpu_number taskset -c range_of_cores ./run_tile.sh
    ```
    
**Options**
```sh
--gpu                    number of the gpu you want to execute the file 
--nr_types               number of categories (nuclei types) 5  
--type_info_path         json with mapping info
--batch_size             size of the batch
--model_mode             'original' 
--model_path             path where the checkpoints are stored 
--nr_inference_workers   8 
--nr_post_proc_workers   16 
tile \
--input_dir              path to test images \
--output_dir             path where the predictions will be stored
--mem_usage              0.1
--draw_dot \
--save_qupath
```

Once you have generate the predictions, you'll have 4 folders inside the folder you indicated to store them. The json folder contains 14 .json files - each corresponding to one image on the testing set - with a dictionary with various information about each nuclei detected (contour, centroid, bbox and type), while qupath contains 14 .tsv files with similar information. The mat folder has the .mat file for each test image, containing the type_map, instance_map, centroids and amount of nuclei detected, and finally, the overlay folder has .png files that show the original image with overlapping boundaries over the detected nuclei.

# Metrics
To be able to compute the metrics associated with the panoptic segmentation performance of each trained model, we have created a main.py file, which present two main arguments.

```sh
--mode       'test' or 'demo'
--img        if mode = 'demo', the number of the image (between 0 and 14) 
```

By choosing 'test' option, you will obtain the metrics for the best model we obtained over all test set, while 'demo' option will show the metrics for a single image you pick.
```sh
python main.py --mode=   --img=
```
# Additional instructions

If you want to train with ResNet50 as backbone, go to run_train.py and make sure the following section in line 203-210 is as shown

```sh
    elif chkpt_ext == "tar":  # ! assume same saving format we desire
        net_state_dict = torch.load(pretrained_path)["desc"]
colored_word = colored(net_name, color="red", attrs=["bold"])
print(
    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
)
net_state_dict = torch.load(pretrained_path)["desc"]
```
If, on the other hand, you want to train with ResNet101, replace the indicated section as follow

```sh
    elif chkpt_ext == "tar":  # ! assume same saving format we desire
        net_state_dict = torch.load(pretrained_path) #["desc"]
colored_word = colored(net_name, color="red", attrs=["bold"])
print(
    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
)
net_state_dict = torch.load(pretrained_path) #["desc"]
```

 If you have any doubts or questions, do not hesitate to contact us through this repository
 
 Enjoy✨


# REFERENCES

```
Graham, S., Vu, Q. D., Raza, S. E. A., Azam, A., Tsang, Y. W., Kwak, J. T., & Rajpoot, N. (2019). Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. Medical Image Analysis, 58, 101563.
```
