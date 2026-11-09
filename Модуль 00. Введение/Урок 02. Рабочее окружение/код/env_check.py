"""Проверка окружения курса: версии библиотек, GPU и реальный прогон ядра на GPU.

Запуск из корня ml_course:  uv run python "Модуль 00. Введение/Урок 02. Рабочее окружение/код/env_check.py"
"""
import importlib
import platform
import sys
import time

PACKAGES = ["numpy", "scipy", "pandas", "matplotlib", "seaborn", "sklearn", "torch", "torchvision"]


def versions() -> dict[str, str | None]:
    """Версия каждого пакета или None, если пакет не установлен."""
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.import_module(name).__version__
        except ImportError:
            out[name] = None
    return out


def gpu_report() -> list[str]:
    """Проверяет не только torch.cuda.is_available(), но и что ядро реально исполняется на GPU."""
    import torch

    if not torch.cuda.is_available():
        return ["CUDA недоступна: проверьте драйвер (nvidia-smi) и что torch собран с CUDA (torch.version.cuda)."]

    lines = []
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    lines.append(f"GPU: {torch.cuda.get_device_name(0)}, compute capability {major}.{minor} ({arch})")
    lines.append(f"torch собран с CUDA {torch.version.cuda}, архитектуры сборки: {torch.cuda.get_arch_list()}")
    if arch not in torch.cuda.get_arch_list():
        # is_available() может вернуть True, а первое же ядро упадёт с "no kernel image is available"
        lines.append(f"ВНИМАНИЕ: сборка не содержит ядер для {arch}. Нужна сборка cu126 (torch<=2.14).")

    free, total = torch.cuda.mem_get_info()
    lines.append(f"Видеопамять: свободно {free / 2**30:.2f} из {total / 2**30:.2f} ГиБ")

    n = 2048
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    try:
        a_gpu, b_gpu = a.cuda(), b.cuda()
        _ = a_gpu @ b_gpu                      # прогрев: первая операция инициализирует CUDA-контекст
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c_gpu = a_gpu @ b_gpu
        torch.cuda.synchronize()               # без synchronize замерили бы только постановку в очередь
        dt = time.perf_counter() - t0
    except RuntimeError as e:
        return lines + [f"ОШИБКА исполнения на GPU: {e}"]

    ok = torch.allclose(c_gpu.cpu(), a @ b, rtol=1e-3, atol=1e-3)
    lines.append(f"matmul {n}x{n} на GPU: {dt * 1e3:.1f} мс, {2 * n**3 / dt / 1e9:.0f} GFLOPS, совпадает с CPU: {ok}")
    return lines


def main() -> None:
    print(f"Python {sys.version.split()[0]} на {platform.system()} {platform.release()}")
    if sys.version_info < (3, 11):
        print("ВНИМАНИЕ: курсу нужен Python 3.11+")
    for name, ver in versions().items():
        print(f"  {name:<12} {ver or 'НЕ УСТАНОВЛЕН'}")
    if versions()["torch"] is not None:
        for line in gpu_report():
            print(line)


if __name__ == "__main__":
    main()
