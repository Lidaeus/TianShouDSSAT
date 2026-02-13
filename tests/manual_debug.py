import os
import shutil
import tempfile
import subprocess

def manual_test():
    tmp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {tmp_dir}")
    
    # 1. 复制 DSSATPRO.L48
    dssat_home = "/home/lidaeus/TianShouDSSAT/lib/gym_dssat_pdi-stable/dssat-csm-data"
    shutil.copy(os.path.join(dssat_home, "DSSATPRO.L48"), os.path.join(tmp_dir, "DSSATPRO.L48"))
    
    # 2. 生成最简 dssat-pdi.yml
    with open(os.path.join(tmp_dir, "dssat-pdi.yml"), "w") as f:
        f.write("pdi: {}\n") 
        
    # 3. 运行 run_dssat
    run_dssat = "/home/lidaeus/TianShouDSSAT/venv/bin/run_dssat"
    
    env = os.environ.copy()
    env["DSSAT_HOME"] = dssat_home
    env["PYTHONPATH"] = "/usr/local/lib/python3/dist-packages"
    
    print("Launching DSSAT...")
    try:
        process = subprocess.Popen(
            [run_dssat, "C", "fileX.MZX", "1"],
            cwd=tmp_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        while True:
            line = process.stdout.readline()
            if not line: break
            print(f"[DSSAT] {line.strip()}")
            if "press < ENTER >" in line.lower():
                print("Detected ENTER request, sending...")
                process.stdin.write("\n")
                process.stdin.flush()
        
        process.wait(timeout=10)
    except Exception as e:
        print(f"Manual test ended: {e}")
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    manual_test()
