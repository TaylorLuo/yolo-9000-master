#! /bin/bash
sum=0
save_path="/media/taylor/G/002---study/rnn_log/output/train_pics/bak/train_pics_01_position/"
#txt_save_path="/media/taylor/G/002---study/rnn_log/output/devkit/cars_train_annos_p1_03/"
txt_save_path="/home/taylor/Documents/homework/yolo-9000-master/darknet/data/obj/"
for file in /home/taylor/Documents/homework/yolo-9000-master/darknet/data/obj/*.jpg
do
    if test -f $file
    then
        let "sum += 1"
        #echo $file if file
        name=${file%.*}                            #去掉.jpg的后缀 /home/dataset/a
        #echo $name
        txtname=$name".txt"                        #加上.txt的后缀 /home/dataset/a.txt
        #echo $txtname
        onlyname=${name##*/}                       #图片的名字a.jpg
        #echo $onlyname
        savename=$save_path$onlyname               #图片保存的路径和名字/home/Yolo_detect_1/a.jpg
        #echo $savename
        #./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights $file -out $savename  #运行检测代码
        ./darknet detect -i 0 cfg/yolo.cfg yolo.weights $file -out $savename -thresh 0.75 #运行检测代码
        mv $txtname $txt_save_path                    #将/home/dataset/a.txt移动到/home/Yolo_detect_1/a.txt
    fi
echo "sum=$sum"
done