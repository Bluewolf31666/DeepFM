# deepFM
deepFM을 deepctr로 구현화 한 코드입니다.
https://deepctr-doc.readthedocs.io/en/latest/Examples.html#classification-criteo
#
computer 가격과 관련된 요소를 feature로 하여 Rating을 예측하는 모델입니다.
#
나중에 파일 수정시에는 preprocessing에서 predict할 label의 존재 유무와 가격과 같이 연속형 데이터의 유무 확인, Sclaer위치 확인이 중요합니다.
(추후 온라인 서비스로 활용하고자 하자면, pickle로 저장해서 불러오거나 하는 방식으로 사용해야함)
