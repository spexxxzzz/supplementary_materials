#!/bin/bash

scancel -u $USER -n loco_kitti
scancel -u $USER -n loco_toy

echo "All LoCo jobs cancelled"
