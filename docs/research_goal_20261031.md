# Research goal (from the user, 2026-10-31)

Uncertainty-Aware Selective In-Context Reasoning with Self-Evolving Memory. The user's text is below, unchanged.

## 1. Remarkable Finding

현재까지의 실험에서 다음과 같은 현상이 관찰되었다.

- **In-Context Sample의 효과는 모든 문항에서 동일하지 않다.** SVLM에 과거 추론 경험을 제공하면 일부 문항에서는 오답이 정답으로 수정되지만, 반대로 기존 정답이 오답으로 바뀌는 경우도 발생한다.
- **모델의 응답 확신도(Confidence)가 낮을수록 In-Context Sample에 의해 답변이 변경되는 경향이 관찰되었다.** 반면 확신도가 높은 문항에서는 불필요한 예시 제공이 기존 정답에 악영향을 미칠 가능성이 나타났다.
- 다만 **답변이 변경되는 것(Answer Sensitivity)과 실제 정답률이 개선되는 것(ICL Effectiveness)은 구분되어야 한다.** 낮은 Confidence 자체가 ICL의 긍정적 효과를 보장하지는 않는다.
- 따라서 SVLM의 응답 불확실성과 ICL 제공에 따른 성능 변화를 함께 분석하면, **ICL이 실제로 도움이 되는 문항을 선택적으로 식별할 가능성**이 있다.

**Core Hypothesis**

> SVLM의 응답 불확실성은 In-Context Sample의 효과와 관련이 있으며, 적절한 예시가 도움이 될 문항에만 선택적으로 개입하면 불필요한 재추론과 성능 저하를 줄이면서 전체 정확도를 개선할 수 있을 것이다.

## 2. Problem Statement

SVLM은 Video Question Answering(VQA) Task에서 제한된 모델 용량으로 인해 복잡한 시각적 정보와 추론 과정을 안정적으로 처리하는 데 어려움을 겪는다.

과거의 유사한 문제와 추론 경험을 In-Context Sample로 제공하면 이러한 한계를 보완할 가능성이 있다. 그러나 **모든 문항에 일률적으로 예시를 제공하는 방식에는 다음과 같은 문제가 존재한다.**

1. **Unnecessary Intervention:** 이미 올바르게 추론한 문항에 추가 예시를 제공하면, 기존 정답이 오답으로 변경될 수 있다.
2. **Uncertain Effectiveness:** 낮은 Confidence를 가진 문항이라도 예시를 제공했을 때 반드시 성능이 개선되는 것은 아니다.
3. **Inference Overhead:** 모든 문항에 예시를 제공하고 재추론하면 불필요한 연산 비용과 추론 지연이 발생한다.
4. **Static Experience Limitation:** 고정된 예시 메모리는 새로운 문제에서 발견되는 실패 유형이나 유용한 추론 경험을 지속적으로 반영하기 어렵다.

따라서 본 연구는 다음 두 가지 질문에 집중한다.

- **When to Intervene:** 어떤 문항에 In-Context Sample을 제공해야 실제 추론 성능이 개선되는가?
- **How to Leverage Experience:** 학생 모델의 과거 추론 경험을 어떻게 축적·개선하고, 필요한 문항에 적절하게 제공할 것인가?

궁극적으로 모든 문항에 예시를 제공하는 방식 대신, **ICL의 기대 유용성을 고려한 선택적 개입(Selective Intervention)** 이 필요하다.

## 3. Proposed Method

본 연구는 **Uncertainty-Aware Selective In-Context Reasoning with Self-Evolving Memory** 프레임워크를 제안한다.

전체 프레임워크는 세 가지 핵심 구성 요소로 이루어진다.

1. **Uncertainty / Utility Estimation:** ICL 개입이 필요한 문항 식별
2. **In-Context Sample Retrieval:** 적절한 과거 추론 경험 검색 및 제공
3. **Self-Evolving Memory:** 학생 모델의 추론 경험을 바탕으로 예시 메모리를 지속적으로 개선

### 3.1. Uncertainty-Aware Selective Intervention

모든 문항에 In-Context Sample을 제공하는 대신, 먼저 학생 모델의 응답 불확실성을 추정하여 개입 여부를 결정한다.

불확실성 측정 방법은 다음 세 가지를 검토한다.

**Method A. Answer Consistency**

- 동일한 문제를 두 번 추론하되, 두 번째 추론에서는 선택지의 순서를 변경한다.
- 두 출력의 답을 원래 선택지 기준으로 변환하여 비교한다.
- 서로 다른 답변을 생성하거나 무효 출력이 발생하면 불확실한 문항으로 판단한다.
- 장점: 모델 내부 확률에 접근하지 않아도 구현 가능하다.
- 단점: 추가적인 추론 비용이 발생하며, 두 답변의 일치가 정답을 보장하지 않는다.

**Method B. Answer Token Confidence**

- 학생 모델의 출력에서 `<ANSWER>`에 해당하는 최종 답변 토큰의 확률을 측정한다.
- 답변 토큰의 확률이 낮을수록 모델의 예측 불확실성이 높다고 가정한다.
- Confidence Threshold를 기준으로 ICL 개입 여부를 결정한다.
- 장점: 추가 생성 없이 모델의 확률 정보를 활용할 수 있다.
- 단점: Token Probability가 실제 정답 확률과 일치하지 않을 수 있으므로 보정 및 검증이 필요하다.

**Method C. Learned ICL Utility Predictor (New)**

- 별도의 데이터에서 학생 모델의 Vanilla 추론과 ICL 추론을 수행하고, 각 문항에서 ICL이 미친 영향을 기록한다.
- Question, Options, Video Embedding 등 추론 전에 확보할 수 있는 특징과 필요시 저비용 모델 신호를 활용해 작은 예측 모델을 학습한다.
- 두 가지 학습 목표를 비교한다.
    - **Confidence Prediction:** 해당 문항의 예상 응답 확신도를 회귀 문제로 학습
    - **ICL Utility Prediction:** ICL 제공이 정답률을 개선할 가능성을 분류 또는 회귀 문제로 학습
- 특히 ICL Utility Prediction은 Confidence 자체보다 **ICL 개입의 기대 이득**을 직접 예측하는 것을 목표로 한다.
- 학생 모델의 Vanilla 생성 결과를 입력으로 사용하지 않는다면, 별도의 선행 생성 없이 ICL 제공 여부를 예측할 수 있다.

최종적으로 세 방법의 성능, 계산 비용, Domain Generalization을 비교하여 선택적 개입 전략을 평가한다.

### 3.2. In-Context Sample Retrieval

ICL 개입이 필요하다고 판단된 문항에 대해서만 과거 추론 경험을 검색한다.

1. 현재 문항의 Video, Question, Options를 임베딩한다.
2. 메모리에 저장된 과거 문항의 임베딩과 Cosine Similarity를 계산한다.
3. 동일 영상의 사례 등 직접적인 정보 누출 위험이 있는 후보를 제외한다.
4. 유사도가 높은 후보 중 Top-1 In-Context Sample을 선택한다.
5. 최고 유사도가 Threshold에 미달하면 예시를 제공하지 않는다.
6. 예시를 제공하는 경우 해당 예시를 Prompt에 삽입하여 학생 모델이 재추론하도록 한다.
7. 재추론 결과가 무효인 경우에는 기존 Vanilla 답변을 유지한다.

Retrieval Strategy, Top-K, Similarity Threshold, Gate Policy는 여러 벤치마크에서 탐색하여 범용적인 설정을 선택한다.

또한 단순한 유사도 검색과 과거 개입 유용도(Utility)를 고려하는 검색 방식을 비교해, **의미적으로 유사한 예시와 실제로 도움이 되는 예시 사이의 차이**를 분석한다.

### 3.3. Self-Evolving In-Context Memory

본 연구는 고정된 예시 집합에 의존하지 않고, 학생 모델의 추론 경험을 바탕으로 지속적으로 업데이트되는 ICL Memory를 구축한다.

**Step 1. Student Inference & Experience Collection**

- 학생 모델이 k번째부터 k+49번째까지 50개 문항을 순차적으로 추론한다.
- 각 문항의 Question, Options, Prediction, Reasoning Trajectory, Confidence, 사용된 ICL Sample을 저장한다.
- 블록의 추론이 완료된 후 정답을 공개하고, 학생의 오류 및 ICL 개입 효과를 기록한다.

**Step 2. Critic Stage 1 — Failure Identification**

- Critic LLM이 학생 모델의 추론 결과와 정답을 비교하여 오류 사례를 분석한다.
- 특히 잘못된 추론 절차로 인해 오답을 출력했을 가능성이 있는 문항을 선별한다.
- 다만 Trajectory와 정답만으로 시각적 인식 실패와 추론 실패를 확실히 구별할 수 없는 경우에는 불확실한 사례로 분류한다.

**Step 3. Critic Stage 2 — Reasoning Experience Generation**

- 선별된 문항에 대해 Teacher Model이 영상과 문제를 바탕으로 새로운 풀이를 생성한다.
- 정답의 근거가 되는 시각적 정보와 추론 과정을 포함하도록 한다.
- 생성된 풀이 중 정확성과 재사용 가능성이 높은 사례를 In-Context Sample Candidate로 선정한다.

**Step 4. Critic Stage 3 — Memory Evolution**

- Critic LLM이 새로운 Candidate와 기존 메모리를 함께 검토한다.
- 다음 네 가지 연산으로 메모리를 업데이트한다.
    - **ADD:** 새로운 추론 경험 추가
    - **MERGE:** 유사하거나 중복된 경험 통합
    - **PRUNE:** 불필요하거나 성능에 악영향을 미치는 경험 제거
    - **REFINE:** 기존 예시의 추론 과정 또는 표현 개선
- Candidate는 과거의 독립된 검증 사례에 대한 효과를 평가한 뒤 채택한다.
- 개별 예시의 실제 사용 실적과 정답 개선·악화 기록을 메모리 관리에 반영한다.

**Step 5. Next-Block Inference**

- 업데이트된 메모리를 이용하여 다음 50문항을 처리한다.
- 각 문항에서 Uncertainty / Utility Estimation을 통해 ICL 필요 여부를 결정한다.
- 개입이 필요한 경우 적절한 예시를 검색해 재추론한다.
- 블록이 완료되면 새로운 경험이 누적되고, 메모리가 다시 갱신된다.

이 과정을 반복함으로써 학생 모델의 파라미터를 업데이트하지 않고도 **과거 추론 경험의 축적과 재사용을 통해 추론 성능을 개선하는 것**을 목표로 한다.
