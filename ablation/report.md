# 실험 결과 보고서
**Lightweight Storage and Offline Reconstruction of Soccer Tracking Data**
Team: doopdoop (DongGwon Kang, GeonHo Kim)

---

## 1. 실험 개요

축구 선수 트래킹 데이터(0.04s 간격)를 절반 해상도(0.08s)로 다운샘플링하여 저장한 뒤,
제거된 중간 프레임을 오프라인으로 복원하는 두 가지 방법을 비교한다.

| 항목 | 내용 |
|---|---|
| 원본 샘플링 | 0.04s (25 Hz) |
| 저장 해상도 | 0.08s (홀수 프레임 보존) |
| 복원 대상 | 짝수 프레임 (Ground Truth) |
| 평가 지표 | RMSE, Discrete Fréchet Distance |

---

## 2. 데이터

- **출처**: DFL Bundesliga Tracking Data (IDSSE Open Dataset)
- **경기 수**: 7경기 (DFL-COM-000001, DFL-COM-000002)
- **궤적 수**: 394개 (선수·심판 FrameSet)
- **전체 샘플**: 11,866,272 쌍 (입력 윈도우 → 중간 프레임)
- **분할**: Train 70% / Val 10% / Test 20%

---

## 3. 방법론

### 3.1 Linear Interpolation (베이스라인)
인접한 두 retained 프레임의 중점으로 추정.

$$\hat{p}_t = \frac{1}{2}(p_{t-1} + p_{t+1})$$

### 3.2 BiLSTM 기반 예측
갭 전후 각 5프레임(총 10프레임)을 컨텍스트로 사용하는 Bidirectional LSTM.

| 하이퍼파라미터 | 값 |
|---|---|
| Hidden size | 128 |
| Layers | 2 (Bidirectional) |
| Context window | 10 프레임 (앞 5 + 뒤 5) |
| Optimizer | Adam (lr=0.001) |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Epochs | 50 |
| Batch size | 512 |

---

## 4. 실험 결과

### 4.1 정량적 평가 (Test set: 2,373,255 샘플)

| Method | RMSE (m) | Fréchet Distance (m) |
|---|---|---|
| **Linear Interpolation** | **0.0732** | **0.0069** |
| LSTM (BiLSTM) | 0.1108 | 0.0308 |

### 4.2 축별 오차 (Mean Absolute Error)

| Method | X-axis (m) | Y-axis (m) |
|---|---|---|
| Linear Interpolation | 0.0044 | 0.0039 |
| LSTM (BiLSTM) | 0.0208 | 0.0178 |

### 4.3 학습 손실

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.9014 | 0.0169 |
| 10 | 0.0119 | 0.0070 |
| 20 | 0.0082 | 0.0061 |
| 30 | 0.0068 | 0.0051 |
| 40 | 0.0035 | 0.0100 |
| 50 | 0.0025 | 0.0048 |
| **Best** | — | **0.0039** |

---

## 5. 분석 및 고찰

**Linear Interpolation이 BiLSTM보다 우수한 이유:**

1. **짧은 시간 간격**: 0.04s 간격에서 선수의 이동 거리는 평균 수 cm 수준으로, 등속도 가정이 매우 잘 성립함.
2. **단순성의 이점**: 복원 대상이 두 retained 프레임 사이의 단일 중간점이므로, 복잡한 시계열 모델보다 단순 보간이 더 안정적.
3. **LSTM의 한계**: BiLSTM은 motion context를 인코딩하나, mid-point representation만으로는 정밀한 위치 추정에 충분하지 않을 수 있음.

---

## 6. 결론

0.08s 간격으로 저장된 트래킹 데이터에서 제거된 0.04s 중간 프레임을
**Linear Interpolation만으로도 RMSE 0.073m 수준으로 복원 가능**함을 확인하였다.

이는 0.04s 전체 스트림을 저장하는 대신 **절반의 저장 공간**으로도 고주파 트래킹의
핵심 통계·운동학적 특성이 보존됨을 시사하며, 대규모 Hadoop 기반 스포츠 데이터
파이프라인에서의 경량 저장 전략으로서 충분히 실용적임을 보인다.

---

## 7. 시각화

| 파일 | 내용 |
|---|---|
| `figures/01_metric_comparison.png` | RMSE / Fréchet 비교 막대그래프 |
| `figures/02_error_distribution.png` | 오차 분포 히스토그램 |
| `figures/03_trajectory_samples.png` | 6개 샘플 궤적 복원 시각화 |
| `figures/04_xy_error.png` | X/Y 축별 MAE 비교 |
