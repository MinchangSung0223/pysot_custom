#!/bin/bash 
mv *.png /home/sung/Yolo_mark_SMC/x64/Release/data/img/
mv *.txt /home/sung/Yolo_mark_SMC/x64/Release/data/img/
cd /home/sung/Yolo_mark_SMC
bash /home/sung/Yolo_mark_SMC/linux_mark.sh
cd /home/sung/SiamMask/data/test
