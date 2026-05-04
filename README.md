# Optics Simulation

## 프로젝트 개요

`optics-simulation`은 실제 PET병 3D STL 형상을 기반으로 광선추적 시뮬레이션을 수행하고, PET병에 의해 형성되는 국부 집광 caustic hot spot을 저감하기 위한 음각 표면 패턴을 설계·최적화·검증하는 연구용 Python 코드베이스입니다.

본 프로젝트의 핵심은 단순히 여러 무늬를 임의로 만들어 비교하는 것이 아닙니다. 실제 PET병 형상에서 빛이 어떻게 굴절되어 hot spot을 형성하는지 분석하고, hot spot 형성에 기여한 표면 영역을 찾아, 그 영역에 표면 법선 분산을 유도하는 원리 기반 음각 패턴을 생성하는 것이 목표입니다.

---

## 연구 목적

본 연구의 최종 목적은 다음과 같습니다.

```text
실제 PET병 STL 형상
        ↓
무패턴 상태 3D ray tracing
        ↓
전체 입사각 스캔
        ↓
집광 위험 각도 구간 탐지
        ↓
hot spot ray 역추적
        ↓
caustic contribution map Rrisk(u, v) 생성
        ↓
Rrisk 기반 음각 dimple pattern 생성
        ↓
표면 법선 분산 증가
        ↓
투과광 각도 분산 증가
        ↓
C99, Eexceed, Ahot, Tmax, dT/dt 저감 검증
```

즉, 본 프로젝트는 PET병에 의해 발생하는 국부 optical/thermal risk indicator를 줄이기 위한 **원리 기반 패턴 생성 알고리즘**을 개발하는 연구입니다.

---

## 연구 범위

본 프로젝트는 다음을 다룹니다.

- 실제 PET병 STL 로드 및 mesh 검증
- 병 중심축 및 표면 정규화 좌표계 정의
- 무패턴 PET병의 3D 광선추적
- 전체 입사각 범위에서 집광 위험 각도 탐지
- hot spot을 만든 ray의 표면 통과 위치 역추적
- caustic contribution map `Rrisk(u, v)` 생성
- `Rrisk` 기반 quasi-random Gaussian dimple pattern 생성
- 패턴 파라미터 최적화
- 적용 전후 optical risk metric 비교
- 간이 thermal risk metric 비교
- 최종 패턴 수식화
- pattern descriptor export
- patterned STL export


본 연구의 주장은 다음 범위로 제한됩니다.

```text
투명 PET병에 의해 형성되는 국부 집광 hot spot과
그로 인한 optical/thermal risk indicator를
표면 패턴 설계를 통해 상대적으로 저감할 수 있는지 검증한다.
```

---

## 핵심 연구 질문

본 연구의 중심 질문은 다음입니다.

> 실제 PET병 STL 형상에서 발생하는 caustic hot spot의 원인 표면 영역을 식별하고, 그 영역에 표면 법선 분산을 유도하는 음각 dimple pattern을 적용하면, 무패턴 병 대비 집광 및 열위험 지표를 유의미하게 낮출 수 있는가?

세부 연구 질문은 다음과 같습니다.

1. 무패턴 PET병은 전체 입사각 범위에서 어떤 각도에서 가장 높은 집광 위험을 보이는가?
2. hot spot을 형성한 ray는 병 표면의 어느 영역을 주로 통과하는가?
3. 해당 표면 영역에 선택적으로 음각 dimple pattern을 적용하면 균일 패턴보다 효과적인가?
4. 표면 법선 분산도 `NDI`가 증가하면 투과광 각도 분산, 즉 `BTDF entropy`가 증가하는가?
5. `BTDF entropy` 증가는 `C99`, `Eexceed`, `Ahot` 감소로 이어지는가?
6. 최종 패턴은 최적화에 사용하지 않은 hold-out 조건에서도 효과를 유지하는가?
7. 최적화로 생성된 패턴을 정규화된 수식과 descriptor로 변환해도 성능이 유지되는가?

---

## 핵심 가설

| 번호 | 가설 | 주요 검증 지표 |
|---:|---|---|
| H1 | 무패턴 PET병은 특정 입사각 구간에서 높은 caustic hot spot을 형성한다. | `C99`, `Eexceed`, `OFRI` |
| H2 | 음각 dimple pattern은 표면 법선 분산도 `NDI`를 증가시킨다. | `NDI` |
| H3 | `NDI` 증가는 투과광 각도 분산 증가로 이어진다. | `BTDF entropy` |
| H4 | 투과광 각도 분산 증가는 caustic peak 감소로 이어진다. | `C99`, `Eexceed`, `Ahot` |
| H5 | caustic contribution map 기반 패턴은 균일 quasi-random 패턴보다 위험 저감 효율이 높다. | `OFRI`, penalty 대비 저감률 |
| H6 | residual correction을 적용한 패턴은 새로 발생하는 residual hot spot에 더 강건하다. | hold-out `OFRI`, `C99`, `Tmax` |
| H7 | 최적화 패턴을 수식화·정규화한 뒤에도 광학·열위험 저감 성능이 허용 오차 내에서 유지된다. | `ΔC99`, `ΔOFRI`, `ΔTmax` |

---

## 제안 패턴의 의미

본 연구에서 말하는 “제안 패턴”은 하나의 고정된 무늬가 아닙니다.

제안 패턴은 다음 요소로 정의되는 **패턴 생성 알고리즘**입니다.

```text
caustic contribution map
        +
quasi-random dimple placement
        +
Gaussian dimple depth field
        +
surface normal dispersion principle
        +
residual hot spot correction
```

최종 패턴은 정규화된 표면 좌표계에서 정의됩니다.

```text
u = 병 둘레 방향 정규화 좌표
v = 병 높이 방향 정규화 좌표
```

음각 깊이 함수는 다음 형태를 기본으로 합니다.

```text
delta(u, v)
= dmax * clip[
    sum_i A_i * exp(-||r - r_i||^2 / (2 * sigma_i^2)),
    0,
    1
]
```

여기서:

| 기호 | 의미 |
|---|---|
| `delta(u, v)` | 표면 위치별 실제 음각 깊이 |
| `dmax` | 최대 허용 음각 깊이 |
| `r = (u, v)` | 정규화 표면 좌표 |
| `r_i` | i번째 dimple 중심 |
| `A_i` | i번째 dimple 깊이 계수 |
| `sigma_i` | i번째 dimple 반경 계수 |

dimple 중심은 완전 랜덤이 아니라 caustic contribution map에 의해 가중됩니다.

```text
P(r_i at u, v) ∝ Rrisk_norm(u, v) + epsilon
```

또한 제조 가능성과 재현성을 위해 최소 간격 조건을 둡니다.

```text
||r_i - r_j|| >= rmin
```

따라서 본 연구의 최종 산출물은 단순 이미지나 STL 변형 결과가 아니라 다음을 포함해야 합니다.

- 정규화 패턴 수식
- pattern descriptor
- dimple center table
- depth field
- patterned STL
- 수식화 전후 성능 검증 결과

---

## 핵심 물리 논리

본 연구는 다음 인과 사슬을 검증합니다.

```text
음각 패턴 적용
        ↓
국소 표면 법선 변화
        ↓
표면 법선 분산도 NDI 증가
        ↓
투과광 출사각 분포 확산
        ↓
BTDF entropy 증가
        ↓
수광면 caustic peak 감소
        ↓
C99, Eexceed, Ahot 감소
        ↓
Tmax, dT/dt 감소
```

중요한 점은 본 연구의 목표가 “빛을 최대한 산란시키는 것”이 아니라는 것입니다.

목표는 다음입니다.

```text
투명도와 제조 가능성을 지나치게 훼손하지 않으면서,
caustic peak를 효과적으로 낮추는 표면 패턴을 찾는 것.
```

---

## 주요 성능 지표

| 지표 | 의미 | 사용 목적 |
|---|---|---|
| `Cmax` | 최대 집광 집중계수 | 보조 지표 |
| `C99` | 상위 1% 조도 평균 기반 집중계수 | 주 optical risk 지표 |
| `Eexceed` | 임계 조도 초과 에너지 | hot spot 에너지 위험 |
| `Ahot` | 임계 조도 이상 영역 면적 | hot spot 면적 |
| `BTDF entropy` | 투과광 각도 분포의 entropy | 각도 분산 지표 |
| `NDI` | 표면 법선 분산도 | 패턴의 물리적 원리 지표 |
| `Tmax` | 수광면 최대 온도 | 열위험 지표 |
| `dT/dt` | 초기 온도 상승률 | 열위험 속도 지표 |
| `OFRI` | 종합 optical/thermal risk indicator | 최적화 목적함수 |

`Cmax`는 ray 수와 detector resolution에 민감할 수 있으므로 주 최적화 지표로 사용하지 않습니다. 본 연구에서는 `C99`, `Eexceed`, `Ahot`, `OFRI`를 더 안정적인 비교 지표로 사용합니다.

---

## 최종 연구 파이프라인

```text
1. PET병 STL 로드
2. mesh 품질 검증
3. 병 중심축 및 표면 좌표계 정의
4. 무패턴 baseline 3D ray tracing
5. 전체 입사각 adaptive scan
6. 위험 각도 구간 Θrisk 탐지
7. hot spot ray 선택
8. ray history 기반 caustic contribution map Rrisk(u, v) 생성
9. Rrisk 기반 Gaussian dimple field 생성
10. 패턴 파라미터 최적화
11. 1차 패턴 적용 후 residual hot spot 분석
12. residual-updated pattern 생성
13. 절차적 optical surface 방식으로 빠른 검증
14. 상위 후보에 대해 실제 STL displacement 생성
15. 전체 각도에서 적용 전후 optical validation
16. 간이 thermal validation
17. hold-out 조건 검증
18. 수식화 및 descriptor export
19. patterned STL export
20. 최종 보고서 생성
```

---

## 비교군

최종 검증에는 다음 비교군을 사용합니다.

| 그룹 | 패턴 유형 | 목적 |
|---|---|---|
| G0 | 무패턴 | 기준 대조군 |
| G1 | 균일 dimple | 단순 음각 효과 확인 |
| G2 | 규칙 grid/groove | 규칙 패턴 비교 |
| G3 | 완전 random dimple | 무작위 패턴 비교 |
| G4 | 균일 quasi-random dimple | 준무작위 효과 분리 |
| G5 | caustic-adaptive quasi-random dimple | contribution map 효과 검증 |
| G6 | residual-updated caustic-adaptive pattern | 잔여 hot spot 보정 효과 검증 |

핵심 비교는 다음입니다.

```text
G0 vs G6: 최종 적용 전후 효과
G4 vs G5: caustic contribution map 사용 효과
G5 vs G6: residual correction 효과
```

---

## 최종 산출물

연구가 진행되면서 다음 산출물을 생성합니다.

```text
clean_bottle.stl
mesh_quality_report.csv
baseline_irradiance_map.npy
angle_response_control.csv
OFRI_vs_angle_control.png
risk_angle_region.json
Rrisk_map.npy
Rrisk_map.png
pattern_descriptor.json
dimple_centers.csv
depth_field.npy
patterned_bottle.stl
before_after_irradiance.png
before_after_temperature.png
G0_to_G6_comparison.csv
holdout_results.csv
formula_validation.csv
final_report.md
```

---

## 개발 원칙

이 프로젝트는 연구용 소프트웨어입니다.

따라서 단순히 코드가 실행되는 것보다 다음이 더 중요합니다.

- 재현 가능성
- 실험 조건 기록
- 물리 가정 명시
- 설정 파일 기반 실험 관리
- 단계별 검증
- Git commit 기반 결과 추적
- 패턴 수식화 및 정규화
- 시뮬레이션 결과의 과장 없는 해석

모든 simulation run은 최종적으로 다음 정보를 저장해야 합니다.

- resolved config
- random seed
- Git commit hash
- input STL hash
- metrics
- generated figures
- run summary