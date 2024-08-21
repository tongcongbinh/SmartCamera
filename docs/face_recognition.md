# FACE RECOGNITION
- Nhận diện khuôn mặt khá chuẩn xác bằng MTCNN và Facenet!
- Chạy trên Tensorflow 2.x
- Article link: http://miai.vn/2019/09/11/face-recog-2-0-nhan-dien-khuon-mat-trong-video-bang-mtcnn-va-facenet/

# TEST
- Train: 
```
python .\src\align_dataset_mtcnn_v2.py .\Data\ .\Data\ --gpu_memory_fraction 0.25
```
```
python .\src\classifier.py TRAIN .\Data\ .\Models\20180402-114759.pb .\Models\AIRC_Face.pkl --batch_size 1000
```
- Chạy mô hình: main.py