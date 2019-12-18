# Machine Learning Perfect Guide Study

- 파이썬 머신러닝 완벽 가이드 스터디

## 스터디 목적

- 머신러닝 애플리케이션 전반에 대한 체계적 복습

## 스터디 방식

1. 인프런 - 파이썬 머신러닝 완벽 가이드 동영상 강의 수강
   - 총 119강, 26시간
2. 파이썬 머신러닝 완벽 가이드 교재로 복습
   - 총 620페이지
3. 깃헙 코드로 3차 복습

## 목차

1. 파이썬 기반의 머신러닝과 생태계 이해 (~p.86)

- Numpy, Pandas

2. 사이킷런으로 시작하는 머신러닝 (~p.142)

- sklearn 프레임워크, Model Selection, 데이터 전처리

3. 평가 (~p.178)

- Accuracy, 오차 행렬, Precision & Recall, Confusion Matrix, F1 Score, ROC/AUC

4. 분류 (~p.285)

- Decision Tree, Ensemble, Random Forest, Gradient Boosting Machine, XGBoost, LightGBM, Under Sampling/Over Sampling, Stacking

5. 회귀 (~p.372)

- Linear Regression, Bias-Variance Trade off, Lidge, Rasso, ElasticNet, Logistic Regression, Regression Tree

6. 차원 축소 (~p.404)

- PCA, LDA, SVD, NMF

7. 군집화 (~p.459)

- K-means, Cluster Evaluation, Mean Shift, GMM, DBSCAN

8. 텍스트 분석 (~p.555)
9. 추천 시스템 (~p.619)

※ 8. 텍스트 분석은 추후 다룰 예정이며, 9. 추천 시스템은 다루지 않을 예정

## 스터디 일지

- 2019.11.07
  - 인프런 - [파이썬 머신러닝 완벽 가이드 강의]( [https://www.inflearn.com/course/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%99%84%EB%B2%BD%EA%B0%80%EC%9D%B4%EB%93%9C#](https://www.inflearn.com/course/파이썬-머신러닝-완벽가이드#) ) 구매
  - [파이썬 머신러닝 완벽 가이드 교재]( http://www.yes24.com/Product/Goods/69752484?scode=032&OzSrank=1 ) 구매
  
- 2019.11.07 ~ 11.08
  - 1장  파이썬 기반의 머신러닝과 생태계 이해 강의 완강, 교재 읽기, 코드 리뷰
    - Numpy, Pandas
  - 머신러닝 기본과 Numpy, Pandas 관련 내용이므로 굳이 할 필요는 없었지만 팁이 될만한 부분을 건질 수 있을 것을 기대하고 빠른 속도로 공부함
  - 데이터 분석시 헷갈릴 만한 것들 교재에 표시해둠
  
- 2019.11.09 ~ 11.10
  - 2장  사이킷런으로 시작하는 머신러닝 강의 완강, 교재 읽기, 코드 리뷰
    - Sklearn 프레임워크, Model selection (K-fold, Stratified K-fold, cross_val_score, GridSearchCV), 데이터 전처리
  - Model Selection에 집중하여 복습함
  
- 2019.11.11 ~ 11.12

  - 3장  평가 강의 완강, 교재 읽기, 코드 리뷰

    - Accuracy, Confusion Matrix, Precison and Recall, F1 Score, ROC/AUC


- 2019.11.13 ~ 11.16

  - 4장  분류 강의 완강, 교재 읽기, 코드 리뷰

    - Decision Tree, Ensemble(voting, bagging, boosting), GBM, XGBoost, LightBoost, Over/Under Sampling(SMOTE), Stacking
  - 모델 학습 코드에 집중하여 복습함
- 2019.11.17 ~ 11.21

  - 5장  회귀 강의 완강, 교재 읽기, 코드 리뷰
  
    - Gradient Descent, Stochastic Gradient Descent, Linear Regression, Polynomial Regression, Regularized Linear Models (Ridge, Lasso, ElasticNet), Logistic Regression, Tree Regression, Preprocessing(Scaling, Log Transformation, Feature Encoding), Mixed Model Prediction
  - 각 휘귀 모델 별 차이점 숙지
  - 스케일링, 인코딩, 아웃라이어 제거, 하이퍼 파라미터 튜닝에 따라 예측 성능이 향상되는 흐름 복습
- 2019.11.22 ~ 11.24
  - 6장  차원 축소
    - 차원 축소 (피쳐 선택, 피쳐 추출), PCA(Principal Component Analysis), LDA(Linear Discriminant Analysis), SVD(Singular Value Decomposition), Truncated SVD, NMF(Non-Negative Matrix Fatorization)
  - 각 차원 축소 기법 별 선형 대수적 의미를 최대한 이해하며 학습
- 2019.11.25 ~ 11.28
  - 7장  군집화
    - K-means, Cluster Evaluation(실루엣 계수), Mean Shift, GMM, DBSCAN
