# RealSense D435 Installation Guide for Raspberry Pi

이 문서는 Raspberry Pi에서 Intel RealSense D435 depth camera를 사용하기 위한 설치 가이드다. Chairshot 프로젝트에서는 `pyrealsense2` Python module을 통해 D435 depth frame을 읽고, `posture/depth_camera.py`와 `develops/realsense_depth_test.py`에서 이를 사용한다.

## 1. Goal

Raspberry Pi에서 다음 항목이 정상 동작하는 것을 목표로 한다.

- D435가 USB 장치로 인식됨
- `rs-enumerate-devices`로 RealSense device 확인 가능
- Python에서 `import pyrealsense2 as rs` 성공
- `develops/realsense_depth_test.py`로 depth frame 수신 가능

## 2. Environment Check

설치 전에 Raspberry Pi에서 아래 정보를 먼저 확인한다. 문제가 생겼을 때 OS, kernel, Python, USB 상태를 구분하기 위해 필요하다.

```bash
cat /etc/os-release
uname -a
python3 --version
which python3
lsusb
lsusb -t
```

D435는 USB 3.x 포트에 연결하는 것이 좋다. `lsusb -t` 결과에서 RealSense 장치가 `5000M`으로 보이면 USB 3.x, `480M`으로 보이면 USB 2.0으로 연결된 것이다. USB 2.0에서도 일부 stream은 가능할 수 있지만 frame drop이나 stream open 실패가 생길 가능성이 크다.

## 3. Fast Check: pip Install

먼저 Python wheel이 현재 Pi 환경과 맞는지 확인한다.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install pyrealsense2
python3 -c "import pyrealsense2 as rs; print('pyrealsense2 ok', rs.__version__)"
```

이 방식이 성공하면 source build가 필요 없다. 하지만 Raspberry Pi OS / Python version / ARM architecture 조합에 따라 wheel이 없거나 import가 실패할 수 있다. 이 경우 source build로 진행한다.

## 4. Source Build Installation

### 4.1 Install Build Dependencies

의존성 누락으로 build가 중간에 실패하는 경우가 많으므로, 먼저 필요한 패키지를 한 번에 설치한다.

2026-05-07 ~ 2026-05-08 설치 로그에서 확인된 패키지와 source build에 일반적으로 필요한 누락 패키지(`git`, `build-essential`, `pkg-config`, `python3-dev`, `python3-numpy`)를 합치면 아래와 같다.

| Category | Packages | Purpose |
| --- | --- | --- |
| Source / build tools | `git`, `cmake`, `cmake-data`, `build-essential`, `pkg-config` | Source clone, CMake configure, compiler/toolchain |
| Core RealSense dependencies | `libssl-dev`, `libusb-1.0-0-dev`, `libudev-dev` | USB device access, udev integration, SSL dependency |
| Python binding | `python3`, `python3-dev`, `python3-pip`, `python3-numpy` | `pyrealsense2` binding build and runtime |
| OpenGL / graphical support | `libgl1-mesa-dev`, `libegl1-mesa-dev`, `libegl-dev`, `libgles-dev`, `libglfw3`, `libglfw3-dev`, `libglvnd-core-dev`, `libglvnd-dev`, `libvulkan-dev` | Viewer, graphical examples, OpenGL/EGL context |
| GTK / windowing support | `libgtk-3-dev`, `libwayland-bin`, `libwayland-dev`, `wayland-protocols` | Graphical UI and Wayland/GTK build dependencies |
| GTK/OpenGL transitive dependencies | `libatk1.0-dev`, `libatk-bridge2.0-dev`, `libatspi2.0-dev`, `libcairo2-dev`, `libepoxy-dev`, `libfontconfig-dev`, `libfreetype-dev`, `libgdk-pixbuf-2.0-dev`, `libglib2.0-dev`, `libpango1.0-dev`, `libxcomposite-dev`, `libxcursor-dev`, `libxdamage-dev`, `libxfixes-dev`, `libxi-dev`, `libxinerama-dev`, `libxkbcommon-dev`, `libxrandr-dev`, `libxrender-dev` | `libgtk-3-dev`, Mesa, GLFW 설치 과정에서 함께 들어오는 development libraries |
| Device permission | `udev` | RealSense udev rule reload/trigger |

Headless 환경에서 `-DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false`로 build한다면 최소 의존성은 아래에 가깝다.

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y \
  git cmake build-essential pkg-config \
  libssl-dev libusb-1.0-0-dev libudev-dev \
  python3 python3-dev python3-pip python3-numpy \
  udev
```

Viewer 또는 graphical example까지 고려한다면 아래 패키지도 추가한다.

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y \
  libgtk-3-dev libglfw3-dev \
  libgl1-mesa-dev libegl1-mesa-dev libegl-dev libgles-dev \
  libglvnd-core-dev libglvnd-dev libvulkan-dev \
  libwayland-bin libwayland-dev wayland-protocols
```

Headless 환경에서 graphical example이 필요 없다면 OpenGL 관련 예제는 build하지 않는 것이 낫다. Raspberry Pi에서는 build 시간이 길고 메모리를 많이 사용하기 때문이다.

### 4.2 Optional: Increase Swap

Pi 4B에서 build 중 memory 부족으로 compiler error가 발생하면 swap을 임시로 늘린다.

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

설치가 끝난 뒤 swapfile을 제거하려면:

```bash
sudo swapoff /swapfile
sudo rm /swapfile
```

### 4.3 Clone librealsense

```bash
cd ~
git clone https://github.com/realsenseai/librealsense.git
cd librealsense
```

기존에 clone한 디렉토리가 있다면:

```bash
cd ~/librealsense
git pull
```

### 4.4 Setup udev Rules

RealSense 장치 접근 권한을 설정한다. 이 단계 전에는 가능하면 D435를 뽑아둔다.

```bash
cd ~/librealsense
./scripts/setup_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

이후 D435를 다시 연결한다.

### 4.5 Build With Python Binding

Raspberry Pi에서는 kernel patch 방식이 OS/kernel 조합에 따라 실패할 수 있다. Chairshot 프로젝트는 depth frame을 Python에서 읽는 것이 핵심이므로, Python binding과 libuvc backend 중심으로 build하는 구성이 현실적이다.

```bash
cd ~/librealsense
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=false \
  -DBUILD_GRAPHICAL_EXAMPLES=false \
  -DBUILD_PYTHON_BINDINGS=true \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DFORCE_LIBUVC=true

make -j$(($(nproc)-1))
sudo make install
sudo ldconfig
```

`$(($(nproc)-1))`가 0이 되는 환경이면 아래처럼 단일 job으로 build한다.

```bash
make -j1
sudo make install
sudo ldconfig
```

### 4.6 Python Path Check

설치 후 Python이 `pyrealsense2`를 찾지 못하면 `/usr/local/lib` 경로를 추가한다.

```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
python3 -c "import pyrealsense2 as rs; print('pyrealsense2 ok', rs.__version__)"
```

매번 적용하려면 `~/.bashrc` 또는 `~/.zshrc`에 추가한다.

```bash
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib' >> ~/.bashrc
source ~/.bashrc
```

환경에 따라 Python binding이 `/usr/local/lib/python3.x/site-packages` 또는 `/usr/local/lib/python3.x/dist-packages` 아래에 설치될 수 있다. 위치를 찾으려면:

```bash
find /usr/local/lib -name 'pyrealsense2*' -print
```

## 5. Verify Installation

### 5.1 Device Enumeration

```bash
rs-enumerate-devices
```

정상이라면 D435 serial number, firmware version, supported stream 정보가 출력된다.

### 5.2 Python Import

```bash
python3 -c "import pyrealsense2 as rs; print('ok', rs.__version__)"
```

### 5.3 Minimal Depth Frame Test

```bash
python3 - <<'PY'
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)
sensor = profile.get_device().first_depth_sensor()
print("depth scale:", sensor.get_depth_scale())

try:
    for _ in range(30):
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth:
            continue
        w, h = depth.get_width(), depth.get_height()
        print("center depth:", depth.get_distance(w // 2, h // 2), "m")
        break
finally:
    pipeline.stop()
PY
```

### 5.4 Chairshot Depth Test

Chairshot 프로젝트 루트에서 실행한다.

```bash
cd ~/chairshot
python3 develops/realsense_depth_test.py
```

OpenCV preview를 보고 싶으면:

```bash
python3 develops/realsense_depth_test.py --preview
```

정상 동작 시 중앙 depth, ROI median depth, valid depth ratio가 출력된다.

## 6. Common Errors

### `ModuleNotFoundError: No module named 'pyrealsense2'`

원인:

- Python binding이 build되지 않음
- `PYTHONPATH`에 설치 경로가 없음
- build한 Python과 실행 중인 Python이 다름

확인:

```bash
which python3
python3 --version
find /usr/local/lib -name 'pyrealsense2*' -print
```

대응:

```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/lib
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
```

### `rs-enumerate-devices`에서 장치가 안 보임

원인:

- USB 케이블 문제
- USB 2.0 포트 연결
- udev rule 미적용
- 전원 부족

확인:

```bash
lsusb
lsusb -t
dmesg | tail -n 80
```

대응:

```bash
cd ~/librealsense
./scripts/setup_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

D435를 분리했다가 다시 연결한다.

### Build 중 dependency 누락

대부분 `cmake` configure 단계에서 어떤 library가 없는지 표시된다. 우선 아래 dependency set을 다시 설치한다.

```bash
sudo apt install -y \
  git cmake build-essential pkg-config \
  libssl-dev libusb-1.0-0-dev libudev-dev \
  python3 python3-dev python3-pip python3-numpy \
  udev
```

그래도 실패하면 에러 로그에서 `Could NOT find ...` 뒤의 package name을 확인해 추가 설치한다.

### Build 중 memory 부족

증상:

- `gcc: internal compiler error`
- build process가 중간에 죽음
- Pi가 멈추거나 SSH가 끊김

대응:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
make -j1
```

### D435 stream start 실패

원인:

- 다른 프로세스가 D435를 사용 중
- USB bandwidth 부족
- 지원하지 않는 stream profile 요청

확인:

```bash
ps aux | grep -i realsense
rs-enumerate-devices
```

낮은 해상도부터 테스트한다.

```bash
python3 develops/realsense_depth_test.py --width 640 --height 480 --fps 30
```

## 7. Information Needed From Your Raspberry Pi

설치 튜토리얼을 더 정확하게 다듬으려면 아래 출력이 필요하다.

```bash
cat /etc/os-release
uname -a
python3 --version
which python3
lsusb
lsusb -t
rs-enumerate-devices
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
dpkg -l | grep -E 'git|build-essential|pkg-config|libusb|libudev|libssl|cmake|glfw|mesa|gtk|wayland|vulkan|python3-dev|python3-numpy'
```

그리고 source build 중 의존성 누락이 발생했던 정확한 에러 메시지가 있으면 아래 항목을 같이 기록한다.

```text
Missing package / CMake error:
Command that failed:
OS:
Kernel:
Python:
librealsense commit or tag:
```

## 8. References

- RealSense SDK repository: https://github.com/realsenseai/librealsense
- Linux installation guide: https://github.com/realsenseai/librealsense/blob/master/doc/installation.md
- LibUVC backend installation: https://github.com/realsenseai/librealsense/blob/master/doc/libuvc_installation.md
- Python wrapper guide: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/readme.md
