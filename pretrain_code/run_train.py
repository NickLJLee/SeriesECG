from __future__ import annotations

import sys


USAGE = """Usage:
  python -m pretrain_code.run_train stage1 --manifest ECG.csv [stage1 args...]
  python -m pretrain_code.run_train stage2 --manifest ECG.csv --label_columns af,bbb [stage2 args...]

This launcher replaces the old hard-coded ECG-CLIP training entrypoint.
The legacy CLIP-style files are still present as train.py/model.py/clip.py.
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"stage1", "stage2"}:
        print(USAGE)
        raise SystemExit(2)

    stage = sys.argv.pop(1)
    if stage == "stage1":
        from pretrain_code.train_stage1 import main as stage_main
    else:
        from pretrain_code.train_stage2 import main as stage_main
    stage_main()


if __name__ == "__main__":
    main()
