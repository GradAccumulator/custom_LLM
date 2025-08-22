import assets.configs.transformer_config as transformer_config
from pathlib import Path
import orjson
from multiprocessing import Pool, cpu_count
from time import perf_counter
from datetime import timedelta

def process_file(p: Path):
    if 'INFO' in p.name:
        return None
    with open(p, "rb") as f:
        data = orjson.loads(f.read())
    results = []
    for paragraph in data["paragraphs"]:
        sentences = paragraph["sentences"]
        if len(sentences) > 1 or (len(sentences) == 1 and sentences[0]["char_count"] > 120):
            text = ''.join(s["text"] for s in sentences)
            if len(text) > 120:
                results.append(text)
    return results

if __name__ == "__main__":
    path = Path(transformer_config.EXTREMELY_RAW_DATA_DIR)
    pathes = list(path.rglob("*.json"))
    total = len(pathes)

    flat_texts = set()
    start_time = perf_counter()

    with Pool(cpu_count()) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, pathes), 1):
            if result:
                flat_texts.update(result)

            # 100개 단위로 ETA 출력
            if i % 100 == 0 or i == total:
                elapsed = perf_counter() - start_time
                per_file = elapsed / i
                remaining = total - i
                eta_seconds = int(per_file * remaining)
                eta = str(timedelta(seconds=eta_seconds))
                print(f"{i} / {total}개 파일 처리 완료 | 예상 남은 시간: {eta}")

    with open(transformer_config.RAW_DATA_PATH, "w", encoding="utf-8") as f:
        f.write('\n'.join(flat_texts))
