#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh 

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal. 
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups. 

# Paths to different tools; 
map_path=/data1/home/hana9000/11775-hws/mAP
export PATH=$map_path:$PATH

# one time thing
python scripts/make_val_labels.py

echo "#####################################"
echo "#       MED with SURF Features      #"
echo "#####################################"
mkdir -p surf_pred
# iterate over the events
feat_dim_surf=225
for event in P001 P002 P003; do
    echo "=========  Event $event  ========="
    # now train a svm model
    python scripts/train_svm.py $event "kmeans/" $feat_dim_surf surf_pred/svm.$event.model || exit 1;
    # apply the svm model to *ALL* the testing videos;
    # output the score of each testing video to a file ${event}_pred 
    python scripts/test_svm.py surf_pred/svm.$event.model "kmeans/" $feat_dim_surf surf_pred/${event}_surf.lst || exit 1;
    # compute the average precision by calling the mAP package
    ap list/${event}_val_label surf_pred/${event}_surf.lst
done

echo ""
echo "####################################"
echo "#    MED with ResNET50 Features    #"
echo "####################################"
mkdir -p resnet50_pred
feat_dim_resnet50=2048
for event in P001 P002 P003; do
    echo "=========  Event $event  ========="
    python scripts/train_svm.py $event "resnet50/" $feat_dim_resnet50 resnet50_pred/svm.$event.model || exit 1;
    python scripts/test_svm.py resnet50_pred/svm.$event.model "resnet50/" $feat_dim_resnet50 resnet50_pred/${event}_resnet50.lst || exit 1;
    ap list/${event}_val_label resnet50_pred/${event}_resnet50.lst
done

echo ""
echo "####################################"
echo "#    MED with PlaceNET Features    #"
echo "####################################"
mkdir -p placesnet_pred
feat_dim_placesnet=4096
for event in P001 P002 P003; do
    echo "=========  Event $event  ========="
    python scripts/train_svm.py $event "places/" $feat_dim_placesnet placesnet_pred/svm.$event.model || exit 1;
    python scripts/test_svm.py placesnet_pred/svm.$event.model "places/" $feat_dim_placesnet placesnet_pred/${event}_placesnet.lst || exit 1;
    ap list/${event}_val_label placesnet_pred/${event}_placesnet.lst
done

echo ""
echo "####################################"
echo "#    MED with HYBRIDNET Features    #"
echo "####################################"
mkdir -p hybrid_pred
feat_dim_hybrid=6144
for event in P001 P002 P003; do
    echo "=========  Event $event  ========="
    python scripts/train_svm.py $event "hybrid/" $feat_dim_hybrid hybrid_pred/svm.$event.model || exit 1;
    python scripts/test_svm.py hybrid_pred/svm.$event.model "hybrid/" $feat_dim_hybrid hybrid_pred/${event}_hybrid.lst || exit 1;
    ap list/${event}_val_label hybrid_pred/${event}_hybrid.lst
done