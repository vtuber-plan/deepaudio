import glob
import os
from typing import Optional

def get_lastest_checkpoint(lightning_logs_path: str, checkpoint_path: str) -> Optional[str]:
    ckpt_path = None
    if os.path.exists(lightning_logs_path):
        versions = glob.glob(os.path.join(lightning_logs_path, "version_*"))
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions))[-1]
            last_ckpt = os.path.join(last_ver, checkpoint_path, "last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    
    return ckpt_path
