# coding = utf-8
import os
import shutil
import argparse
import csv
from tqdm import tqdm 


def copy_file(infile, outdir, key_name: str = "wav_path"):
    os.makedirs(outdir, exist_ok=True)
    if infile.endswith('.csv'):
        # 先把所有行读出来，方便统计总数
        with open(infile, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        for row in tqdm(reader, desc="Copying files", unit="file"):
            if key_name in row:
                raw_path = row[key_name]
                if os.path.isfile(raw_path):
                    filename = os.path.basename(raw_path)
                    dst_path = os.path.join(outdir, filename)
                    shutil.copy2(raw_path, dst_path)
                else:
                    tqdm.write(f"Warning: file not found -> {raw_path}")
    else:
        print(f"Unsupported file type: {infile}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    args = parser.parse_args()

    infile = args.infile
    outdir = args.outdir
    copy_file(infile, outdir)
    