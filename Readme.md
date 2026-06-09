# README

## 1. Intro

### Project Summary

Chairshot은 사용자의 앉은 자세를 실시간으로 감지하고, 자세 상태를 LED와 웹 대시보드로 피드백하는 Raspberry Pi 기반 posture monitoring project이다. Raspberry Pi AI Camera(IMX500)는 pose estimation으로 코, 좌/우 어깨 등 핵심 keypoint를 추적하고, Intel RealSense D435는 keypoint/ROI 주변 depth 값을 측정한다. 시스템은 초기 바른 자세를 baseline condition으로 저장한 뒤, 이후 측정값이 baseline 대비 얼마나 변했는지를 기준으로 `normal`, `turtle_neck`, `shoulder_tilt`, `out_of_range` 상태를 판별한다.

### Goals

- 사용자가 장시간 앉아 있을 때 자세 변화를 실시간으로 감지한다.
- 거북목과 좌우 어깨 기울기를 구분해 bad posture로 판단한다.
- LED를 통해 현재 상태를 즉시 피드백한다.
- Flask + SQLite 대시보드 서버에 안정적으로 유지된 자세 상태만 기록한다.
- 센서 처리와 백엔드 저장을 분리해 Raspberry Pi 클라이언트는 posture detection에 집중한다.

### Used Hardware

| Hardware | Role |
| --- | --- |
| Raspberry Pi 4B | Main runtime device |
| Raspberry Pi AI Camera(IMX500) | Pose estimation, keypoint detection |
| Intel RealSense D435 | Depth measurement |
| Green LED(GPIO 18) | Normal posture |
| Yellow LED(GPIO 23) | Baseline / out of range |
| Red LED(GPIO 24) | Bad posture alert |

### Tech Stack

| Area | Stack |
| --- | --- |
| Camera / Vision | Picamera2, IMX500 HigherHRNet pose estimation, OpenCV |
| Depth | pyrealsense2, RealSense D435 depth frame |
| Hardware Control | gpiozero, Raspberry Pi GPIO |
| Client Runtime | Python, threaded sensor workers |
| Backend | Flask |
| Database | SQLite3 |
| Communication | HTTP POST event publishing |

### Setup Guides

- [Intel RealSense D435 installation on Raspberry Pi](./realsense%20installation.md)

## 2. Body

### Quick Start

Backend server:

```bash
python3 flask_server.py --host 0.0.0.0 --port 8800
```

Client with preview:

```bash
python3 posture_monitor.py \
  --event-url http://127.0.0.1:8800/api/posture-events/
```

Client without preview:

```bash
python3 posture_monitor.py \
  --no-preview \
  --event-url http://127.0.0.1:8800/api/posture-events/
```

If the backend runs on another machine, replace `127.0.0.1` with the backend server IP.

### User Flow With LED

1. 사용자가 의자에 바르게 앉은 상태에서 `posture_monitor.py`를 실행한다.
2. 시스템은 `baseline_seconds=5.0`초 동안 유효한 pose/depth frame을 수집한다.
3. Baseline 설정 중에는 green/yellow/red LED가 모두 점등된다.
4. Baseline이 완료되면 실시간 posture monitoring이 시작된다.
5. 현재 상태가 `normal`이면 green LED가 점등된다.
6. 현재 상태가 `turtle_neck` 또는 `shoulder_tilt`이면 bad posture로 분류되고 red LED가 점등된다.
7. 사용자가 인식 범위를 벗어나거나 keypoint/depth 값이 유효하지 않으면 `out_of_range`로 분류되고 yellow LED가 점등된다.
8. LED 상태 전환은 순간 오탐을 줄이기 위해 같은 상태가 `led_switch_delay_seconds=1.0`초 유지된 뒤 적용된다.
9. Bad posture는 `bad_duration_seconds=5.0`초 이상 지속될 때 `alarm=True`가 된다.
10. Flask 대시보드로 전송되는 posture event는 같은 상태가 `event_publish_stable_seconds=5.0`초 이상 유지된 뒤부터 기록된다.

### Bad Posture Conditions

#### Common Baseline

Baseline은 최초 바른 자세에서 5초 동안 수집된 pose/depth 값의 평균 또는 median이다. Pose baseline은 head/shoulder 위치와 어깨 각도를 저장하고, depth baseline은 `head`, `shoulder`, `chest`, `nose` depth 값을 저장한다. 이후 모든 posture 판단은 현재 frame과 baseline의 delta를 비교해서 수행한다.

| Parameter | Current Value | Meaning |
| --- | ---: | --- |
| `baseline_seconds` | 5.0s | 기준 자세 수집 시간 |
| `min_keypoint_confidence` | 0.25 | nose, left shoulder, right shoulder 최소 confidence |
| `bad_duration_seconds` | 5.0s | bad posture alarm 발생 기준 지속 시간 |
| `bad_recovery_tolerance_seconds` | 1.0s | bad 상태 중 짧은 normal 흔들림을 무시하는 시간 |
| `out_of_range_tolerance_seconds` | 0.5s | 인식 이탈 확정 전 허용 시간 |
| `led_switch_delay_seconds` | 1.0s | LED 상태 변경 debounce |
| `event_publish_stable_seconds` | 5.0s | DB 저장 전 상태 안정화 시간 |

#### Shoulder Tilt

Shoulder tilt는 좌/우 어깨 keypoint의 높이 차이와 baseline 대비 어깨 라인 각도 변화량으로 판단한다. 단순 pixel 차이만 사용하면 카메라 사선 각도에서 오탐이 커지기 때문에, 어깨 라인 각도 변화량을 함께 사용한다.

| Condition | Threshold |
| --- | ---: |
| 어깨 라인 각도 변화량 | `>= 12.0deg` |
| 또는 어깨 높이 차이 변화량 + 각도 변화량 조합 | `shoulder_tilt_delta_px >= 22.0px` and `shoulder_angle_delta >= 8.0deg` |

판정 결과는 `reason=shoulder_tilt`이며, red LED 대상 bad posture이다.

#### Turtle Neck

Turtle neck은 머리/코가 baseline보다 앞으로 이동했지만 어깨 라인은 비교적 안정적인 경우로 판단한다. 현재 구현은 head depth, shoulder depth stability, shoulder angle stability를 가중치로 합산한 score 방식과 nose depth delta 보조 조건을 함께 사용한다.

Depth delta 정의:

| Delta | Formula | Meaning |
| --- | --- | --- |
| `head_delta_m` | `baseline_head_depth - current_head_depth` | 양수면 머리가 카메라 쪽으로 가까워짐 |
| `shoulder_delta_m` | `baseline_shoulder_depth - current_shoulder_depth` | 양수면 어깨가 카메라 쪽으로 가까워짐 |
| `nose_delta_m` | `current_nose_depth - baseline_nose_depth` | 음수면 코가 카메라 쪽으로 가까워짐 |

Weighted turtle neck score:

| Component | Threshold / Formula | Weight |
| --- | ---: | ---: |
| Head forward score | `head_delta_m / 0.07m` | 0.60 |
| Shoulder stable score | `1 - abs(shoulder_delta_m) / 0.05m` | 0.25 |
| Shoulder angle stable score | `1 - shoulder_angle_delta / 12.0deg` | 0.15 |
| Final score threshold | `>= 0.65` | - |

Additional nose-depth condition:

| Condition | Threshold |
| --- | ---: |
| Shoulder angle stable | `< 12.0deg` |
| Nose moved forward | `nose_delta_m <= -0.05m` |
| Shoulder depth stable | `abs(shoulder_delta_m) <= 0.015m` |

위 weighted score 조건 또는 nose-depth 조건 중 하나를 만족하면 `reason=turtle_neck`으로 판정한다. 판정 결과는 red LED 대상 bad posture이다.

#### Out Of Range

Out of range는 사용자가 카메라 인식 범위를 벗어나거나, 핵심 keypoint/depth 측정값이 유효하지 않을 때 발생한다.

| Case | Condition |
| --- | --- |
| Person missing | pose keypoint가 검출되지 않음 |
| Keypoint confidence low | nose/left shoulder/right shoulder confidence가 `0.25` 미만 |
| Missing depth | `head`, `shoulder`, `chest` ROI 중 하나 이상의 median depth가 없음 |
| Tolerance | 위 상태가 `0.5s` 이상 지속되면 `out_of_range` 확정 |

판정 결과는 `state=out_of_range`이며 yellow LED가 점등된다.

#### Normal

Normal은 `turtle_neck`, `shoulder_tilt`, `out_of_range` 조건을 만족하지 않는 상태이다. 단, 직전에 bad posture가 감지된 경우 `bad_recovery_tolerance_seconds=1.0`초 이내의 짧은 normal 흔들림은 무시하고 이전 bad 상태를 유지한다.

### Current Runtime Architecture

| Module | Responsibility |
| --- | --- |
| `posture_monitor.py` | Sensor orchestration, baseline flow, logging/event publishing |
| `posture/imx_camera.py` | IMX500 AI Camera control and pose callback |
| `posture/workers.py` | PoseWorker and DepthWorker threaded processing |
| `posture/pose.py` | Keypoint selection and ROI extraction |
| `posture/depth.py` | ROI/keypoint depth measurement |
| `posture/analyzer.py` | Baseline and posture condition classification |
| `posture/alerts.py` | LED state control |
| `posture/event_publisher.py` | HTTP POST event publishing |
| `flask_server.py` | Flask dashboard API and SQLite persistence |
