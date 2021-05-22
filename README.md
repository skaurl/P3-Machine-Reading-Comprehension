# P3-Machine-Reading-Comprehension

# 전체 개요 설명

"한국에서 가장 오래된 나무는 무엇일까?" 이런 궁금한 질문이 있을 때 검색엔진에 가서 물어보신 적이 있을텐데요, 요즘엔 특히나 놀랍도록 정확한 답변을 주기도 합니다. 어떻게 가능한 걸까요? 질의 응답(Question Answering)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다. 그 중에서도 Open-Domain Question Answering 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가가 되어야하기에 더 어려운 문제입니다.

![fc81eeb5-3cc0-44b6-a5dc-aaf38e4166a5](https://user-images.githubusercontent.com/55614265/116175274-f2ed7980-a74a-11eb-83c8-bba40f25b77e.png)

본 대회에서 우리가 만들 모델은 두 stage로 구성되어 있습니다. 첫 번째 단계는 질문에 관련된 문서를 찾아주는 "retriever"단계이고요, 다음으로는 관련된 문서를 읽고 간결한 답변을 내보내 주는 "reader" 단계입니다. 이 두 단계를 각각 만든 뒤 둘을 이으면, 어려운 질문을 던져도 척척 답변을 해주는 질의응답 시스템을 여러분 손으로 직접 만들게 됩니다. 더 정확한 답변을 내주는 모델을 만드는 팀이 우승을 하게 됩니다.

![cf0936da-a81a-4ea0-897d-4441d12c402a](https://user-images.githubusercontent.com/55614265/116324692-88484680-a7fb-11eb-9de7-8ac98ef69622.png)

# 평가 방법

평가방법은 두가지입니다.

1. Exact Match (EM): 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어집니다. 즉 각 질문마다 0점 아니면 1점입니다. 다만 띄어쓰기나 "."과 같은 문자가 포함돼 있다고 오답으로 처리되면 억울하겠죠? 이런 것은 지우고 일치하는지 확인합니다. 또한 답이 하나가 아닐 수 있는데, 이런 경우는 하나라도 일치하면 정답으로 간주합니다.

![1](https://user-images.githubusercontent.com/55614265/116324960-0efd2380-a7fc-11eb-8ba8-32d4de379ee9.png)


2. F1 Score: EM과 다르게 부분 점수를 제공합니다. 예를 들어, 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만 F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 줍니다.

![2](https://user-images.githubusercontent.com/55614265/116324976-14f30480-a7fc-11eb-8388-a822ae07c15a.png)

EM 기준으로 랭킹을 산정하고, F1은 참고용으로만 활용합니다.

# 학습 데이터 개요

다음은 제공하는 데이터셋의 구성을 보여줍니다.

![8acf9df5-ea23-4ad2-a24c-c9a847d24e59](https://user-images.githubusercontent.com/55614265/116324051-52569280-a7fa-11eb-991a-9de3531df2be.png)

MRC 데이터의 경우, Hugging Face에서 제공하는 datasets library로 접근이 가능합니다. 해당 폴더의 directory를 dataset_name 으로 저장한 후, 아래와 같이 불러오기가 가능합니다.

```
# train_dataset을 불러오고 싶은 경우
from datasets import load_from_disk
dataset = load_from_disk("./data/train_dataset/")
print(dataset)
```

Retrieval 과정에서 사용하는 문서 집합(corpus)은 ./data/wikipedia_documents.json 으로 저장되어있습니다. 약 5만 7천개의 unique 한 문서로 이루어져 있습니다. 평가 데이터는 학습데이터와 대부분 동일하나, 리더보드용인 test_dataset 의 데이터에는 id 와 question 만 주어집니다. 즉 Open Domain QA 전용이며, 답을 내기 위해서 ./data/wikipedia_documents.json 을 활용합니다. 데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```
./data/         # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성
    ./dummy_dataset/           # 모델이 작동하는지 테스트 할때 사용하는 dummy 데이터셋. train 과 validation 으로 구성
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

- id: 질문의 고유 id
- question: 질문
- answers: 답변에 대한 정보. 하나의 질문에 하나의 답변만 존재함
- answer_start : 답변의 시작 위치
- text: 답변의 텍스트
- context: 답변이 포함된 문서
- title: 문서의 제목
- document_id: 문서의 고유 id

# 평가 데이터 개요

평가 데이터는 학습데이터와 대부분 동일하나, 리더보드용인 testdataset 의 데이터에는 id 와 question 만 주어집니다. 즉 Open Domain QA 전용이며, 답을 내기 위해서 ./data/wikipediadocuments.json 을 활용합니다. 대회가 종료된 후에는 한 번도 보지 못했던 테스트 데이터를 기반으로 최종 등수가 결정됩니다.

# 코드 설명
```
$> tree -d
.
```

# Wrap up Report

## 기술적인 도전

### [Github Repo](https://github.com/bcaitech1/p3-mrc-tajo)

### 본인의 점수 및 순위
- Public : LB 점수 EM: 66.25%, F1: 78.70%, 2등
- Private : LB 점수 EM: 64.17%, F1: 74.88%, 2등

### EDA
- [토론](http://boostcamp.stages.ai/competitions/31/discussion/post/254)
- [Colab](https://colab.research.google.com/drive/1xgla_ghhOlbjDqAWYPieQxNs7nIRnHd1)

### 기본적인 전략
- EDA를 통해서 해당 Competition에서는 Retrieval가 중요하다고 판단.
- 따라서 리더보드에 제출했을 때, Retrieval의 성능 만을 확인하기 위해서는 일정한 성능을 보장하는 MRC가 필요.
- 이에 적합한 PORORO의 MRC를 선택하여 Retrieval 성능을 빠르게 실험 그리고 제출 기회를 최대한으로 활용.
- 해당 정보를 바탕으로 팀 공용 Retrieval Setting 및 팀원의 모델 학습에 도움.

### 검증(Validation)전략
- 제공된 Train dataset과 Validation dataset을 사용하지 않았기 때문에, 해당 데이터 전부를 Validation으로 사용.

### Retrieval
![](https://images.velog.io/images/skaurl/post/2e9c6f7b-a58b-4be1-a5c0-50de2cf84cc3/%EA%B7%B8%EB%A6%BC1.png)
- 해당 세팅은 최종 버전이며, 이외에도 시도해볼 수 있는 모든 세팅에 대해서 테스트를 진행.
- 해당 세팅 과정에서 본의 아니게 엘라스틱 서치에 대해서 많은 것을 배울 수 있었음. (가이드북 정독, 관련 서적 정독)

![](https://images.velog.io/images/skaurl/post/7f315f44-35dc-433d-9cac-81bf3b439cde/%EA%B7%B8%EB%A6%BC2.png)
- 세팅 이외에도 쿼리를 어떻게 주느냐에 따라서 엘라스틱 서치의 성능이 천차만별.
- 여러 번의 시도 끝에, NER을 통해서 선정된 단어에 가중치를 방식으로 최종 결정.

### PORORO MRC
- PORORO의 MRC를 그대로 사용하기에는 두 가지 문제점이 존재
    - 문제점 1. Logit(Score)를 출력하지 않음.
    - 문제점 2. Top-1의 Answer 만을 출력.
- 소스 코드를 수정하여 위 두가지 문제를 해결.
- 해당 해결 과정에서 나온 코드를 PORORO에 PR할 계획.

### 앙상블 방법
- Rule base Hard Voting
- 단순하게 Hard Voting을 하기에는 후처리 방법에 따라서 그 결과가 달라짐.
- 따라서 후처리 뿐만 아니라 Rule Base 기반의 알고리즘을 추가하여 앙상블을 시도함.
- 또한, 각기 다른 모델들의 장점을 잘 살릴 수 있도록 앙상블을 설계.
- 해당 방법 덕분에 마지막 날 6% 이상의 성능을 상승.

### 시도했으나 잘 되지 않았던 것들
1. rank_bm25 라이브러리에 미구현 되어 있는 BM25-Adpt와 BM25T를 구현하려고 했으나, 공식에 대한 이해 부족으로 실패.
2. Hugging Face와 Haystack 라이브러리를 기반으로 한국어 Dense Passage Retrieval을 시도했으나 실패.
    - 양질의 Train 데이터 부족
    - 높은 배치사이즈를 위한 컴퓨팅 파워 부족
    - 한국어로 pretrain된 모델의 부재
3. KoBART를 기반으로 하는 Generation-based MRC를 구현
    - 후처리를 안해도 된다는 장점이 존재
    - 단, Extraction-based MRC의 성능을 이기지 못함
4. 엘라스틱 서치의 다양한 세팅을 통한 Retrieval ensemble
5. Doc2Vec Embedding을 이용한 Retrieval
6. Bert Embedding을 이용한 Retrieval
7. Pyserini 라이브러리 사용

## <학습과정에서의 교훈>

### 학습과 관련하여 개인과 동료로서 얻은 교훈
- 개인으로 진행하는 competition과 팀으로 진행하는 competition의 전략은 달라야 한다는 것을 배웠습니다.
- 특히, 팀으로 진행하는 competition의 경우 더욱 더 많은 실험을 해볼 수 있기 때문에, 정보 공유와 취합 그리고 정리가 중요합니다.
- 그리고 대회 막바지 서로 다른 모델들이 존재하기 때문에, 어떻게 앙상블을 할지 미리 생각해야 할 것 같습니다.
- 끝으로, 팀을 위해 개인이 어디까지 희생해야 하는지 그리고 개인의 이득을 위해서 어디까지 팀을 이용해야 하는지 그 밸런스가 중요한 것 같습니다.
- 이를 위해서는 확실한 리더의 존재가 필요하다고 생각합니다.
 
### 피어세션을 진행하며 좋았던 부분과 동료로부터 배운 부분
- 피어세션을 하면서 가장 좋았던 부분은 두 가지가 있습니다.
- 첫째, 다른 팀원들의 실험 내용을 참고할 수 있었습니다.
- 둘째, 제 실험 내용을 설명하기 위해 스스로 한 번 정리한다는 것입니다.

## <마주한 한계와 도전숙제>

### 아쉬웠던 점들
- 저는 이번 스테이지에서 1등을 하고 싶었습니다.
- 하지만 저희 팀은 졌고 2등으로 대회를 마무리했습니다.
- 1등을 하기 위해서 최선을 다했던 것 만큼, 가장 아쉬운 부분입니다.

### 한계/교훈을 바탕으로 다음 스테이지에서 새롭게 시도해볼 것
- 그냥. 1등 할 생각입니다. 다른 것은 잘 모르겠습니다.
