# Notebook에서 얻어낸 것들.

| Topic | Describe |
| :---: | :------: |
| [Ensemble Modeling](##ensemble-modeling) | Ensemble modeling 순서와 Cross Validation 시각화 |
| [Learning Curve](##learning-curve) | Learning Curve를 시각화하여 '과적합'과 'train size의 영향' 확인하기 |

---

## :computer: Ensemble Modeling

### [Ensemble modeling 순서]

```
1. Cross Validation을 이용한 model performance 확인
    > ensemble modeling에 활용할 model 선정

2. Grid Search를 이용한 Hyperparameter tunning
    > 각 model에 대해 best model을 생성

3. best model들을 결합하여 ensemble modeling
```

### [Cross Validation]

> 교차 검증 결과를 시각화하여 model을 선정하자!

1. 검증할 model들을 list에 append

```python
random_state = 2
classifiers = []

classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))

...
```

2. 각 model들을 iteration 하면서 `cross_val_score()` 함수로 교차 검증

```python
cv_results = []

for classifier in tqdm(classifiers) :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring="accuracy", cv = kfold, n_jobs=1))
```

3. 교차 검증 socre의 평균과 표준편차 계산

```python
cv_means = []
cv_std = []
for cv_result in tqdm(cv_results):
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
```

4. 계산한 score의 평균과 표준편차를 Algorithm 별로 DataFrame화

```python
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree", ... ]})
```

5. 평균 값을 기준으로 dataframe을 정렬 (가장 높은 점수부터 보기 위해서)

```python
cv_res = cv_res.sort_values(by="CrossValMeans")
```

6. **교차검증 점수의 평균값을 기준으로 시각화**

```python
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
```
---

## :chart_with_upwards_trend: Learning Curve

> 함수를 작성하여 활용하자!

1. 함수 정의부

```python
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
```
    - learning_curve() 함수에 사용하는 parameter
        - estimator
        - X, y
        - cv
        - n_jobs
        - train_sizes

2. plt 기본 figure 설정

```python 
plt.figure() # 새로운 figure 생성
plt.title(title) # figure 제목 붙이가

if ylim is not None:
    plt.ylim(*ylim)

plt.xlabel("Training examples")
plt.ylabel("Score")

plt.grid() # 그래프에 grid 표시되게!
```

3. model_selection의 `learning_curve()`함수를 이용해서 'train_sizes', 'train_scores', 'test_scores' 계산

```python
train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
```

4. 'train socre'와 'test score'에 대해 mean, std 값 계산

```python
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
```

5. 각 learning curves에서 std 만큼의 영역 색 채우기

```python
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
```

6. figure에 그래프 그리기

```python
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
```

7. figure 반환

```python
plt.legend(loc="best")

return plt
```

8. 시각화 함수 사용하기

```python
g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
```

---

