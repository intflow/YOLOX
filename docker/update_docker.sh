#!/bin/bash

sudo docker commit yolox yolox:dev_1.0_30xx_ubuntu18.04
sudo docker login docker.io -u kmjeon -p 1011910119a!
sudo docker tag yolox:dev_1.0_30xx_ubuntu18.04 intflow/yolox:dev_1.0_30xx_ubuntu18.04
sudo docker push intflow/yolox:dev_1.0_30xx_ubuntu18.04

