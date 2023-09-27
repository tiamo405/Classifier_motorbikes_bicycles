# 1. Data
```
data    __train __xedap_0000.jpg
                __.....
        __valid     
        __test 
```
- dowload data
```
chua cap nhat
```
# 2. Train
```
Cài đặt các thư viện như pytorch, numpy, opencv-python, tdqm, torchvision ...
Khuyến khích sử dụng anaconda
```

```
python train.py --data_root data/train --name_model resnet101 --checkpoint_dir checkpoints
```
- folder weight :
```
checkpoint_dir __ name_model __ times train __ number epoch.pth
ex : checkpoints/resnet101/0001/1.pth
```
# 3. Test

```
python predict.py --checkpoint_dir checkpoints --name_model resnet101 --num_train 0001 --num_ckpt 1 --image data/test/image.jpg
``` 
* Test on colab
```
--checkpoint_dir : 
```
## Trên drive lưu file weight như sau:
```
drive/weight/DandC/resnet101/0001/1.pth
```
[Dowload pretrained](https://drive.google.com/)
# 4.Colab
[Colab demo](https://colab.research.google.com/)