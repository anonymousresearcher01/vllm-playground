# CUDA Error 804: Forward Compatibility on Non-Supported HW

## Overview

본 문서는 `nvidia-smi`는 정상 동작하지만 CUDA 애플리케이션 (PyTorch, vLLM 등) 이 아래 에러와 함께 실패하는 경우 대처할 수 있는 트러블슈팅 가이드이다.

```
RuntimeError: Unexpected error from cudaGetDeviceCount().
Error 804: forward compatibility was attempted on non supported HW
```

---

## Environment

| Component | Version |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 3060 Ti |
| OS | Ubuntu 20.04 |
| Kernel | 5.4.0-216-generic |
| NVIDIA Driver (nvidia-smi) | 535.230.02 |
| CUDA Version (nvidia-smi) | 12.2 |
| PyTorch CUDA build | 12.1 |



## Diagnosis Steps

### Step 1. Python/torch 레벨 배제

torch 를 완전히 배제하고 CUDA 드라이버 API 를 직접 호출해서 결과값을 확인한다.

```python
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
result = libcuda.cuInit(0)
print(f'cuInit result: {result}')
# 0   = CUDA_SUCCESS
# 804 = CUDA_ERROR_FORWARD_COMPATIBILITY_NOT_SUPPORTED
```

`cuInit`은 CUDA 드라이버 API의 가장 낮은 레벨 초기화 함수이고, 여기서 804가 반환되면 Python/torch/vLLM 은 원인이 아닌 것으로 판단할 수 있다.

### Step 2. 드라이버 버전 일치 여부 확인

```bash
# nvidia-smi 버전
nvidia-smi | grep "Driver Version"

# 커널 모듈 버전
cat /proc/driver/nvidia/version
modinfo nvidia | grep "^version"

# dkms 빌드 버전
dkms status | grep nvidia
```

세 버전이 모두 일치해야 한다.

### Step 3. libcuda.so 심볼릭 링크 확인

```bash
ls -la /lib/x86_64-linux-gnu/libcuda*
ls -la /usr/lib/x86_64-linux-gnu/libcuda*
```

아래와 같이 링크 대상 버전이 커널 모듈 버전과 다르면 안된다.

```
libcuda.so.1 -> libcuda.so.535.261.03   ← 커널 모듈(535.230.02)과 불일치
libcuda.so.535.230.02
libcuda.so.535.261.03
```

### Step 4. strace 로 로드 경로 추적

위 단계에서 명확하지 않을 경우 strace 로 확인합니다.

```bash
strace -e trace=openat python -c "
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
print(libcuda.cuInit(0))
" 2>&1 | grep nvidia
```

출력에서 로드되는 `libcuda.so` 버전과 application profile 경로의 버전 번호를 확인한다.

---

## Solution

커널 모듈 버전에 맞는 라이브러리로 심볼릭 링크를 수정함으로써 해결할 수 있다.

```bash
# 커널 모듈 버전 확인
modinfo nvidia | grep "^version"
# → 535.230.02

# 심볼릭 링크 수정
sudo ln -sf libcuda.so.535.230.02 /lib/x86_64-linux-gnu/libcuda.so.1
sudo ln -sf libcuda.so.535.230.02 /usr/lib/x86_64-linux-gnu/libcuda.so.1

# 검증
python -c "
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
result = libcuda.cuInit(0)
print(f'cuInit result: {result}')  # 0 이면 성공
import torch
print('cuda available:', torch.cuda.is_available())
"
```

---

## Why This Happens

드라이버 업그레이드 시 패키지 관리자 (apt) 가 유저스페이스 라이브러리는 교체하지만, DKMS 커널 모듈 재빌드가 실패하거나 재부팅 없이 종료되면 두 컴포넌트 간 버전 불일치가 발생합니다. GeForce 시리즈 GPU 는 NVIDIA forward compatibility 기능을 지원하지 않기 때문에, 유저스페이스와 커널 모듈 버전이 다르면 `cuInit`이 Error 804를 반환한다.

## Prevention

드라이버 업그레이드 후에는 반드시 재부팅하고, 아래 명령으로 버전 일치를 확인해야 한다.

```bash
# 한 번에 버전 비교
echo "=== nvidia-smi ===" && nvidia-smi | grep "Driver Version"
echo "=== kernel module ===" && modinfo nvidia | grep "^version"
echo "=== libcuda.so.1 ===" && ls -la /usr/lib/x86_64-linux-gnu/libcuda.so.1
echo "=== cuInit test ===" && python -c "
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
print('cuInit:', libcuda.cuInit(0))
"
```

---

## References

- [NVIDIA CUDA Compatibility Documentation](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- NVIDIA Error Code 804: `CUDA_ERROR_FORWARD_COMPATIBILITY_NOT_SUPPORTED`