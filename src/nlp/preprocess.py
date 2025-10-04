from pathlib import Path
import PATHES
from multiprocessing import (
    Pool,
    cpu_count
)
import pandas as pd


config = {
    'data_file_suffix': 'parquet',
    'max_workers': cpu_count()-1,
    'data_folder': PATHES.ASSETS_BIGDATA_EXTREMELY_RAW/'huggingface'/'hub'/'datasets--HAERAE-HUB--KOREAN-WEBTEXT'/'snapshots'/'data',
    'save_path': PATHES.ASSETS_BIGDATA_RAWS/'korean_webtext_raw.txt'
}

def preprocess(data_file:Path)->list[str]:
    return pd.read_parquet(data_file)['text'].tolist()


def preprocess_ko_text() -> None:
    raw_files = list(Path(config['data_folder']).glob('*.'+config['data_file_suffix']))
    
    with Pool(config['max_workers']) as pool:
        result = pool.map(preprocess, raw_files)
    
    print("writing to a file...")
    with open(config['save_path'], 'w', encoding='utf-8') as f:
        for text in result:
            for line in text:
                f.write(line.replace('\n', '')+"\n")

if __name__ == '__main__':
    print("start preprocessing corpus")
    preprocess_ko_text()
    print("complete!!")
