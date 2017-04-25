#!/usr/bin/env bash

# Any time you use the Mac file explorer to view a directory, it adds a file '.DS_Store' that saves the view
# preferences of the user.  It's a hidden file, so whatever.  Except it throws off a lot of file operations
# in this project.  Simply run this script to remove all the .DS_Store files in the data directory.

find /centurion/FREDcast -name '.DS_Store' -type f -delete

# Copy relevant programs to deep learning servers
scp -r ~/PycharmProjects/FREDcast/ rgio@cobalt.centurion.ai:/centurion/FREDcast/

# rsync -r ~/PycharmProjects/FREDcast/ rgio@fireball.cs.uni.edu:/storage/xbrl_mapping/

scp -r ~/PycharmProjects/FREDcast/ rgio@buff.centurion.ai:/centurion/FREDcast/

scp -r ~/PycharmProjects/FREDcast/ /centurion/FREDcast/