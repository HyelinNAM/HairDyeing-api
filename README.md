# HairDyeing-api

main.py를 통해 염색 이미지를 생성할 수 있는 방법은 2가지 입니다.

아래 방법을 통해 이미지를 생성하기 위해서는 [Face parsing](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view), [SEAN](https://drive.google.com/file/d/1UMgKGdVqlulfgOBV4Z0ajEwPdgt3_EDK/view)의 pretrained 모델을 다운 받아, pretrained_network 폴더에 담아주세요.

```python
pip install -r requirements.txt
```
위 명령어를 통해 필요한 패키지들을 설치할 수 있습니다.


## 1. Reference 이미지 이용 </br>
: 머리 색을 가져오고싶은 reference 이미지가 있는 경우

```python
1) data/ref, data/src 폴더 생성

2) data/ref 폴더에는 reference 이미지를, data/src 폴더에는 source 이미지를 저장

3) python main.py --mode custom 
```


## 2. 미리 저장된 색으로 염색 </br>
: pkl 파일(Python 3.8 이상에서만 가능)에 미리 저장된 색을 이용해 염색하고자 하는 경우 (지원 색상 - black, brown, blond, red, blue)

```python
1) data/src 폴더 생성

2) data/src 폴더에 source 이미지 저장

3) python main.py --mode {color_name}
   ex. python main.py --mode black
```

+pkl 파일 색상 코드 변경 또는 새로운 색상 추가하기
```python
1) data/ref 폴더 생성

2) data/ref 폴더에 reference 이미지 저장

3) python main.py --mode save_color --color_name {color_name} 
   ex. python main.py --mode save_color --color_name pink
```

## Reference
- [SEAN](https://github.com/ZPdesu/SEAN)
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
