汽车检测项目
实现二：

首先
记录一下，yolo9000没有成功实现，虽然用了yolo9000的配置文件，但最终实现时yolo_v2的方法：
1.https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects 中有说9000的配置文件，但没有清楚的描述wordtree应该是一个什么样的关系，个人推测是
　9k.tree中应该是车辆的764个类型，如果combine9k.data中map = data/coco9k.map,则对应父类都是3802
2.比较坑的是9k.names中不能有中文，目前未找到解决办法，只能用数字代替，然后再在显示的时候想办法对应回来
3.然后就是obj中图片对应txt文件中，boundingbox信息到底写不写，尝试过程中，如果不写则训练失败，不知道这个地方到底怎么处理，https://github.com/pjreddie/darknet/issues/450中
介绍的方法，写上boundingbox信息，是可以训练成功的，本文的实现方式就是用了这种方法，但不是9000的正宗方法．
4.如果用AlexeyAB给出的方法，则9k.tree中对应的父类需要写成-1，即认为这些类需要重新检测边框，不能依照coco或者inet已有经验进行检测

然后
记录一下，因为现有数据的无边框问题，使用yolo_v2的前期准备工作：

0.按照现有状况，手里只有tfrecord文件，并且知道里面的标注信息没有边框信息，所以我的做法是发扬不折腾不舒服斯基的精神：

解析tfrecord文件，获得image，接着用ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb检测到边框，写出jpg和对应的边框信息txt文件
脚步：reGenerateTfrecord.py

文件存放目录：
/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_011～041
/media/taylor/G/002---study/rnn_log/output/validation_pics/validation_pics_011～041
/media/taylor/G/002---study/rnn_log/output/devkit/devkit/cars_train_annos_p1_01.txt～04.txt
/media/taylor/G/002---study/rnn_log/output/devkit/cars_validation_annos_p1_01.txt～04.txt


1.在获取边框信息的问题上，刚开始还想着用yolo快速获取边框，使用get_pic_position.sh脚本，能够得到原始图片和每个图片对应一个文本文件来记录边框信息，
但发现虽然速度快（不到１秒一张图片，mobilenet的话，大约3秒一张），但检测效果没有mobilenet好，所以放弃了这种方法
该弃用方法的实现步骤:(鄙人尝试写英文，是不是有国民亲切感．．．)
First: in detector.h line 705 add a method like draw_detections named draw_detections_car,and add one more parameter named char *filename
        to transfer the name of pic
Second: in image.c line 311 to add the implementation of draw_detections_car

Third: in detector.c line 615 to replace draw_detections with draw_detections_car

Finally: make

then, ．/get_pic_position.sh


2.通过步骤0已经获取到了所有数据，就用之前第八周作业的方式使用faster-rcnn进行训练，但是效果久久不能出来，不知道原因．试想既然图片数据已经获得，并且比compcars正宗原始数据格式要简单，
就尝试一下yolo9000．在尝试yolo9000时，没有相关文档，但最终还是放弃了，文章开头已经记录，此处不再赘述．
然后就尝试用yolo_v2的训练方法，跑yolo9000的配置：
在实现过程中，发现obj文件目录中要求存放图片原始文件和每张相对应的相同名称的txt文件，该txt文件标明图片中物体所属类别和边框信息：e.g. 1 0.716797 0.395833 0.216406 0.147222
are center of rectangle (are not top-left corner) 这个坐标不同于常用的基于左上的坐标，他是基于中心位置的坐标
做到这一步的时候，我欣喜的发现步骤１中可以实现给每张图分别产生一个txt文件，还带有边框信息，只需改一下边框计算方法就ＯＫ了，所以所以修改了image.c，使其产生文件，虽然１秒张图，但是１秒也是
时间啊，所以放弃该方法，该方法测试目录/home/taylor/Documents/homework/yolo-9000-master/darknet/data/testcars

因为在步骤0中已经得到了所有信息，所以直接改成读取cars_train_annos_p1_01.txt，进行解析，然后分别生成对应文件
其中0.3535040020942688	0.08965688943862915	0.7543930411338806	0.9647945761680603	582	18_Label_582.jpg
坐标信息是通过mobilenet的pb，一个个预测出来的，效果还不错
但是yolo中的坐标是基于中心点的，而不是上面的基于左上的方式，所以要转换，所以写了一个脚本getImage-txt.py
并在网络上找了一个文件，可供参考voc_label.py


3.有了obj下的图片和标注信息，有了yolo9000的6个文件：9k.labels, 9k.names, 9k.tree, coco9k.map, combine9k.data, yolo9000.cfg，以及train.txt, test.txt
接下来就是训练了


4.car-detection-yolo_v2训练方法：

 get pre-trained file

./darknet partial data/yolo9000.cfg ../yolo9000-weights/yolo9000.weights ../yolo9000-weights/yolo9000.conv.22 22

./darknet detector train data/combine9k.data data/yolo9000.cfg ../yolo9000-weights/yolo9000.conv.22

detect:
./darknet detector test data/combine9k.data data/yolo9000.cfg backup/yolo9000_80000.weights

./darknet detector test data/combine9k.data data/yolo9000.cfg /media/taylor/H/CSDN/study-nets/yolo/backup/yolo9000_140000.weights

/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_011/0_Label_676.jpg
/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_011/1_Label_546.jpg
/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_011/3_Label_656.jpg
/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_011/25_Label_132.jpg
/media/taylor/G/002---study/rnn_log/output/train_pics/train_pics_011/1466_Label_28.jpg

/home/taylor/Pictures/选区_001.jpg
/home/taylor/Pictures/选区_001.jpg

预测图片存放目录：/home/taylor/Documents/homework/yolo-9000-master/darknet

继续训练：
./darknet detector train data/combine9k.data data/yolo9000.cfg /media/taylor/H/CSDN/study-nets/yolo/backup/yolo9000_140000.weights


遗留问题：9k.names的中文支持问题
参考：
https://blog.csdn.net/xueruixuan/article/details/78840846
https://blog.csdn.net/hrsstudy/article/details/61671886
https://blog.csdn.net/babala258/article/details/78161408





5.DEMO:https://github.com/pjreddie/darknet/issues/450 训练方法
./darknet partial data/yolo9000_air.cfg ../yolo9000-weights/yolo9000.weights ../yolo9000-weights/yolo9000.conv.22_air 22

./darknet detector train data/air9k.data data/yolo9000_air.cfg ../yolo9000-weights/yolo9000.conv.22_air

detect:
./darknet detector test data/air9k.data data/yolo9000_air.cfg /media/taylor/H/CSDN/study-nets/yolo/backup-9000/yolo9000_air_200.weights -thresh .9
./darknet detector test data/air9k.data data/yolo9000_air.cfg /media/taylor/H/CSDN/study-nets/yolo/backup-9000/yolo9000_air_100.weights /media/taylor/G/002---study/aiProject/yolo-9000-51ai/darknet/data/obj/0_Label_676.jpg -thresh .1
./darknet detector test data/air9k.data data/yolo9000_air.cfg /media/taylor/H/CSDN/study-nets/yolo/backup-9000/yolo9000_air_900.weights /media/taylor/G/002---study/aiProject/yolo-9000-51ai/darknet/data/obj/1_Label_546.jpg -thresh .15
/media/taylor/G/002---study/aiProject/yolo-9000-51ai/darknet/data/obj/0_Label_676.jpg
/media/taylor/G/002---study/aiProject/yolo-9000-51ai/darknet/data/obj/1_Label_546.jpg

继续训练：
./darknet detector train data/air9k.data data/yolo9000_air.cfg /media/taylor/H/CSDN/study-nets/yolo/backup-9000/yolo9000_air_100.weights



# Yolo 9000
YOLO9000: Better, Faster, Stronger - Real-Time Object Detection (State of the art)

<p align="center">
  <img src="img/example.gif" width="500"><br/>
  <i>Scroll down if you want to make your own video.</i>
</p>

## How to get started?

### Ubuntu/Linux
```
git clone --recursive https://github.com/philipperemy/yolo-9000.git
cd yolo-9000
cat yolo9000-weights/x* > yolo9000-weights/yolo9000.weights # it was generated from split -b 95m yolo9000.weights
md5sum yolo9000-weights/yolo9000.weights # d74ee8d5909f3b7446e9b350b4dd0f44  yolo9000.weights
cd darknet 
make # Will run on CPU. For GPU support, scroll down!



./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights data/horses.jpg

./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights data/00013.jpg


./darknet detector test data/combine9k.data data/yolo9000.cfg ../yolo9000-weights/yolo9000.weights data/horses.jpg



Yolo-v2用法

./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg

./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights data/dog.jpg

```

### Mac OS
```
git clone --recursive https://github.com/philipperemy/yolo-9000.git
cd yolo-9000
cat yolo9000-weights/x* > yolo9000-weights/yolo9000.weights # it was generated from split -b 95m yolo9000.weights
md5 yolo9000-weights/yolo9000.weights # d74ee8d5909f3b7446e9b350b4dd0f44  yolo9000.weights
cd darknet 
git reset --hard b61bcf544e8dbcbd2e978ca6a716fa96b37df767
make # Will run on CPU. For GPU support, scroll down!
./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights data/horses.jpg
```

You can use the latest version of `darknet` by running this command in the directory `yolo-9000`:

```
git submodule foreach git pull origin master
```

The output should be something like:

```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   544 x 544 x   3   ->   544 x 544 x  32
    1 max          2 x 2 / 2   544 x 544 x  32   ->   272 x 272 x  32
    2 conv     64  3 x 3 / 1   272 x 272 x  32   ->   272 x 272 x  64
    3 max          2 x 2 / 2   272 x 272 x  64   ->   136 x 136 x  64
    4 conv    128  3 x 3 / 1   136 x 136 x  64   ->   136 x 136 x 128
    5 conv     64  1 x 1 / 1   136 x 136 x 128   ->   136 x 136 x  64
    6 conv    128  3 x 3 / 1   136 x 136 x  64   ->   136 x 136 x 128
    7 max          2 x 2 / 2   136 x 136 x 128   ->    68 x  68 x 128
    8 conv    256  3 x 3 / 1    68 x  68 x 128   ->    68 x  68 x 256
    9 conv    128  1 x 1 / 1    68 x  68 x 256   ->    68 x  68 x 128
   10 conv    256  3 x 3 / 1    68 x  68 x 128   ->    68 x  68 x 256
   11 max          2 x 2 / 2    68 x  68 x 256   ->    34 x  34 x 256
   12 conv    512  3 x 3 / 1    34 x  34 x 256   ->    34 x  34 x 512
   13 conv    256  1 x 1 / 1    34 x  34 x 512   ->    34 x  34 x 256
   14 conv    512  3 x 3 / 1    34 x  34 x 256   ->    34 x  34 x 512
   15 conv    256  1 x 1 / 1    34 x  34 x 512   ->    34 x  34 x 256
   16 conv    512  3 x 3 / 1    34 x  34 x 256   ->    34 x  34 x 512
   17 max          2 x 2 / 2    34 x  34 x 512   ->    17 x  17 x 512
   18 conv   1024  3 x 3 / 1    17 x  17 x 512   ->    17 x  17 x1024
   19 conv    512  1 x 1 / 1    17 x  17 x1024   ->    17 x  17 x 512
   20 conv   1024  3 x 3 / 1    17 x  17 x 512   ->    17 x  17 x1024
   21 conv    512  1 x 1 / 1    17 x  17 x1024   ->    17 x  17 x 512
   22 conv   1024  3 x 3 / 1    17 x  17 x 512   ->    17 x  17 x1024
   23 conv  28269  1 x 1 / 1    17 x  17 x1024   ->    17 x  17 x28269
   24 detection
Loading weights from ../yolo9000-weights/yolo9000.weights...Done!
data/horses.jpg: Predicted in 7.556429 seconds.
wild horse: 50%
Shetland pony: 84%
Aberdeen Angus: 72%
Not compiled with OpenCV, saving to predictions.png instead
```

The image with the bounding boxes is in `predictions.png`. 

## Examples

`./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights data/horses.jpg`
<div align="center">
  <img src="img/predictions_horses.png" width="400"><br><br>
</div>

`./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights data/person.jpg`
<div align="center">
  <img src="img/predictions_person.png" width="400"><br><br>
</div>

Browse on https://pjreddie.com/darknet/yolo/ to find how to compile it for GPU as well. It's much faster!

## GPU Support

Make sure that your NVIDIA GPU is properly configured beforehand. `nvcc` should be in the PATH. If not, *something like this* should do the job:

```
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```
Let's now compile `darknet` with GPU support!
```
cd darknet
make clean
vim Makefile # Change the first two lines to: GPU=1 and CUDNN=1. You can also use emacs or nano!
make
./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights data/dog.jpg
```

The inference should be much faster:
```
Loading weights from ../yolo9000-weights/yolo9000.weights...Done!
data/dog.jpg: Predicted in 0.035112 seconds.
car: 70%
canine: 56%
bicycle: 57%
Not compiled with OpenCV, saving to predictions.png instead
```

You can also run the command and monitor its status with `nvidia-smi`:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.26                 Driver Version: 375.26                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN X (Pascal)    Off  | 0000:02:00.0      On |                  N/A |
| 26%   49C    P2    76W / 250W |   4206MiB / 12189MiB |     10%      Default |
+-------------------------------+----------------------+----------------------+
|   1  TITAN X (Pascal)    Off  | 0000:04:00.0     Off |                  N/A |
| 29%   50C    P8    20W / 250W |      3MiB / 12189MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:05:00.0     Off |                  N/A |
| 31%   53C    P8    18W / 250W |      3MiB / 12189MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  TITAN X (Pascal)    Off  | 0000:06:00.0     Off |                  N/A |
| 29%   50C    P8    22W / 250W |      3MiB / 12189MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0     30782    C   ./darknet                                     3991MiB |
+-----------------------------------------------------------------------------+
```
Here, we can see that our process `darknet` is running on the first GPU.

**NOTE**: We highly recommend a recent GPU with 8GB (or more) of memory to run flawlessly. GTX 1070, GTX 1080 Ti or Titan X are a great choice!

## Make your own video! (Ubuntu/Linux)

First we have to install some dependencies (OpenCV and ffmpeg):
```
sudo apt-get install libopencv-dev python-opencv ffmpeg
cd darknet
make clean
vim Makefile # Change the first three lines to: GPU=1, CUDNN=1 and OPENCV=1. You can also use emacs or nano!
make
./darknet detector demo cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights  -prefix output <path_to_your_video_mp4> -thresh 0.15
```
By default the threshold is set to 0.25. It means that Yolo displays the bounding boxes of elements with a 25%+ confidence. In practice, a lower threshold means more detected items (but also more errors).

Once this command returns, we merge the output images in a video:
```
ffmpeg -framerate 25 -i output_%08d.jpg output.mp4
```

We can now safely remove the temporary generated images:
```
rm output_*.jpg
```

The final video is `output.mp4`.

## Important notes

It was successfully tested on Ubuntu 16.04 and Mac OS. I had it working on MacOS with a previous version of `darknet`. I now get a SEGFAULT on the newest `darknet` version with MacOS El Capitan. That's the reason why I pulled a slightly older version of `darknet` for Mac OS.

