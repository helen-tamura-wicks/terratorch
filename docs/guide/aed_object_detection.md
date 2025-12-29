# what happens when you run terratroch fit -c <config.yaml> for the aed object detection 
## 1- you run the command
## 2- config file is read
## 3- classes are created (Datamodule, Task, Trainer)
## 4- dataset (tiling) happens
## 5- dataloader batches are formed
## 6- model is built (framework + backbone)
## 7- training loop starts (ft)
## 8- outputs are checkpoints, logs, and  tile cache
