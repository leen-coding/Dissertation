1）file_process.py 用于生成待匹配人物图像的地址列表，结果保存为name_list.txt
2）sunglasses文件夹里面是墨镜模板的图片，经过人工处理，对于背景和边界都做了一些调整
3）sunglasses_annotation文件夹里是墨镜对应的人工标注信息，对应code可见annotation_main.cpp
4）batch_process_test是批量生成数据的代码，依赖库是dlib和opencv3，运行方式：
mkdir build
cd build
cmake ..
make
cp -i ../shape_predictor_68_face_landmarks.dat .
./main
5）dlib-19.9zip是dlib库的源码，编译安装方式如下：
mkdir lib
unzip dlib-19.9.zip 
cp -r dlib-19.9 lib/ 
cd lib/dlib-19.9/
mkdir dlib_build
cd dlib_build/
cmake ..
cmake --dlib_build .
--------------------
注意：batch_process_test对应的Cmakelist需要根据实际环境进行修改