
```sed -i 's#/home/bykang/voc#home/chris/Fewshot_Detection/safari#g' data/voc_traindict_full.txt```

https://stackoverflow.com/questions/2099471/add-a-prefix-string-to-beginning-of-each-line

https://github.com/marvis/pytorch-yolo2/issues/84


https://vxlabs.com/2018/11/04/pytorch-1-0-preview-nov-4-2018-packages-with-full-cuda-10-support-for-your-ubuntu-18-04-x86_64-systems/

For CUDA version 10.1
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

For CUDA version 10.0
From <https://github.com/nagadomi/waifu2x/issues/253#issuecomment-445448928>:
```
git clone https://github.com/nagadomi/distro.git ~/torch --recursive
cd ~/torch
./install-deps
./clean.sh
./update.sh
```

sed -i 's#0#2#g' data/voc_traindict_full.txt

## Create data for training
### step 1 Renaming files to make sure conversion scripts capture all file names
#change the patterns in rename_files and run:
python3 rename_files.py
#in img and ann
for f in *\ *; do mv "$f" "${f// /_}"; done
### step 2 Convert supervisely format to Pascal VOC
#change the patterns in supervisely_to_pascal_voc and run:
python3 supervisely_to_pascal_voc.py
#in the new PascalVoc format folder, run the command below for each folder:
#Annotations, labels, and JPEGImages
cd Annotations
for f in * ; do mv -- "$f" "synth_$f" ; done
cd ../labels
for f in * ; do mv -- "$f" "synth_$f" ; done
cd ../JPEGImages
for f in * ; do mv -- "$f" "synth_$f" ; done
### step 3 Create and mv files for training
cd ..
mkdir labels_1c
cp -r labels labels_1c/bird
#in labels
cd labels
sed -i 's#0 #2 #g' *.txt
cd ../../CanistersRealTrainVal
#for i in JPEGImages Annotations labels labels_1c/bird/
#do
#rm ${i}/train* 
#done

for i in JPEGImages Annotations labels labels_1c/bird/
do
cp -r ../Canisters3740/${i}/can3740* ${i}/.
done

for i in JPEGImages Annotations labels labels_1c/bus labels_1c/cow labels_1c/motorbike labels_1c/sofa
do
cp -r ../CanistersNotAnn/${i}/can* ${i}/.
done

for name in test*
do
    newname=train"$(echo "$name" | cut -c5-)"
    mv "$name" "$newname"
done

## Training
python2 train_meta.py cfg/metatune.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg backup/000150.weights

exp=15
rm -r ../safariland-element/CanistersRealTrainVal/annotations_cache/
python2 valid_ensemble.py cfg/metatune.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg backup/metatunetest${exp}_novel0_neg0/000020.weights /home/chris/Fewshot_Detection/data/voc_traindict_bbox_10shot-${exp}.txt
rm preds/*
python3 test_image.py ${exp}
rm preds${exp}.zip
zip preds${exp}.zip -r preds
python2  scripts/voc_eval.py results/metatunetest${exp}_novel0_neg0/ene000020/comp4_det_test_


./darknet detect ../Fewshot_Detection/cfg/darknet_dynamic.cfg ../Fewshot_Detection/backup/metatunetest12_novel0_neg0/000010.weights test1.jpg

python2 detect.py cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg  backup/metatunetest15_novel0_neg0/000010.weights ../safariland-element/CanistersRealTest/JPEGImages/test_00000.jpg

Test Sets
http://localhost:8887/edit/safariland-element/val_can.txt
http://localhost:8887/edit/safariland-element/CanistersRealTrainVal/ImageSets/Main/test.txt