#!/usr/bin/env python3
"""Download the Linux wheels for the cluster on your own PC, in parallel.

Only downloads .whl files into wheels_linux/ - installs nothing, touches no env.
Then copy them to the cluster:

    python bdx_mjx/slurm/download_wheels_pc.py            # 8 at a time
    python bdx_mjx/slurm/download_wheels_pc.py --jobs 12
    scp wheels_linux/* 23me36008@login08.iitkgp.ac.in:/scratch/scratch26/23me36008/bdx_env/wheels/

Re-running skips packages already in wheels_linux/.
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, "..", "requirements-lock-linux.txt")

# Target = the cluster container: Linux x86_64, CPython 3.11, glibc up to 2.36.
PLATFORM_ARGS = ["--python-version", "3.11", "--implementation", "cp",
                 "--abi", "cp311", "--abi", "abi3", "--abi", "none"]
for g in range(12, 37):
  PLATFORM_ARGS += ["--platform", f"manylinux_2_{g}_x86_64"]
PLATFORM_ARGS += ["--platform", "manylinux2014_x86_64", "--platform", "manylinux2010_x86_64"]


def already_have(spec, dest):
  name, version = spec.split("==")
  prefix = f"{name.replace('-', '_').lower()}-{version}-"
  return any(f.lower().startswith(prefix) for f in os.listdir(dest))


def download(spec, dest):
  t = time.time()
  cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
         "--disable-pip-version-check", "--quiet", "--timeout", "120", "--retries", "10",
         "-d", dest, *PLATFORM_ARGS, spec]
  r = subprocess.run(cmd, capture_output=True, text=True)
  return spec, r.returncode, time.time() - t, r.stderr.strip().splitlines()[-1:] if r.stderr else []


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--jobs", type=int, default=8)
  p.add_argument("--dest", default="wheels_linux")
  args = p.parse_args()
  os.makedirs(args.dest, exist_ok=True)

  specs = [l.strip() for l in open(LOCK) if l.strip() and not l.startswith("#")]
  todo = [s for s in specs if not already_have(s, args.dest)]
  print(f"{len(specs)} packages, {len(specs) - len(todo)} already downloaded, "
        f"{len(todo)} to go, {args.jobs} at a time")

  failed, start = [], time.time()
  for attempt in (1, 2, 3):  # failures (timeouts) get retried automatically
    if attempt > 1:
      if not failed:
        break
      print(f"\nretrying {len(failed)} failed package(s), attempt {attempt}/3")
      todo, failed = failed, []
    with cf.ThreadPoolExecutor(args.jobs) as pool:
      for i, fut in enumerate(cf.as_completed(pool.submit(download, s, args.dest) for s in todo), 1):
        spec, rc, secs, err = fut.result()
        size = sum(os.path.getsize(os.path.join(args.dest, f)) for f in os.listdir(args.dest))
        status = "done  " if rc == 0 else "FAILED"
        print(f"[{i}/{len(todo)}] {status} {spec:45s} {secs:6.1f} s   total {size / 1e9:.2f} GB")
        if rc != 0:
          failed.append(spec)
          print("        ", *err)

  print(f"\nfinished in {time.time() - start:.0f} s, {len(os.listdir(args.dest))} files")
  if failed:
    print("failed (re-run to retry):", " ".join(failed))
    sys.exit(1)
  print("now: scp wheels_linux/* 23me36008@login08.iitkgp.ac.in:"
        "/scratch/scratch26/23me36008/bdx_env/wheels/")


if __name__ == "__main__":
  main()
