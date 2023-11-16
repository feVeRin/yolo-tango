# YOLOv7 NAS for TANGO Framework

## Members
* [Wonseon Lim](https://github.com/laymond1)
* [Heon-Sung Park](https://github.com/hopo55)
* [Sang-Hyeong Jin](https://github.com/feVeRin)   
* [Jaegyun Park](https://github.com/jgpark92)
* [Hanyong Lee](https://github.com/lhy0718)

## Goal

[YOLOv7](https://github.com/WongKinYiu/yolov7)을 Base Architecture로 Neural Architecture Search(NAS), specifically [Once-for-All Paper](https://github.com/mit-han-lab/once-for-all), 방식으로 Scalable YOLOv7 버전을 만드는 것이 목표.

## TODO List

- [X] ELAN Block 정의
- [X] YOLO SuperNet 정의
- [X] Evolution Search 구현 (실행 파일: search_yolo.py)
- [X] 모바일 NAS 코드 통합
- [X] TANGO Repo 양식으로 변환 및 통합
- [X] 코드 정제 
  - [X] 각 코드 파일 별 불필요한 라이브러리 로드 제거
  - [X] 모듈 형식 통일
  - [X] README 작성
  

## Idea 정리

* __Control the number of layers(branch) in each an E-ELAN block__

    : __각 E-ELAN block 내에서 layer의 분기 수를 조절하는 방식__

* TANGO Framework에서 neckNAS > etri directory에 있는 코드 구조 참고