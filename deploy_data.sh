#!/usr/bin/env bash

rsync -u /centurion/FREDcast/*hdf5 rgio@fireball.cs.uni.edu:/centurion/FREDcast/

rsync -u /centurion/FREDcast/*lrz rgio@fireball.cs.uni.edu:/centurion/FREDcast/

rsync -u /centurion/FREDcast/*hdf5 rgio@cobalt.centurion.ai:/centurion/FREDcast/

rsync -u /centurion/FREDcast/*lrz rgio@cobalt.centurion.ai:/centurion/FREDcast/

rsync -u /centurion/FREDcast/*hdf5 rgio@10.0.1.14:/centurion/FREDcast/

rsync -u /centurion/FREDcast/*lrz rgio@10.0.1.14:/centurion/FREDcast/