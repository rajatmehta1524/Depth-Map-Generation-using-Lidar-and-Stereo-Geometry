#!/bin/bash

python generate_disparities.py --loadmodel trained_model.tar

python dump_test.py --weight_path ./pretrained_weights.pth --data_path ./test_depth_completion_anonymous --dump