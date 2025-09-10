# pipeline_bootstrap.py
from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import polars as pl
import requests
import contextlib



# ==============================
# Config / paths
# ==============================

BASE_URLS = [
    "https://dadosabertos.rfb.gov.br/CNPJ/dados_abertos_cnpj/",
    "https://arquivos.receitafederal.gov.br/dados/cnpj/dados_abertos_cnpj/",
]

USER_AGENT = "cnpj-cnae-pipeline/3.0"
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 60

RAW_ROOT = Path("data/raw")                         # downloads here: data/raw/<YYYY-MM>/*.zip
TMP_ROOT = Path("data/tmp_estab_extracts")          # temp extraction of CSV/TXT
DDBB_ROOT = Path("data/ddbb_scd")
WORKING_DIR = DDBB_ROOT / "working"                 # <YYYY-MM>_pairs.parquet
OPEN_DIR = DDBB_ROOT / "open"                       # open_intervals.parquet
HIST_DIR = DDBB_ROOT / "history"                    # history_closed.parquet
PARQUET_COMPRESSION = "zstd"


HEADERS = {"User-Agent": USER_AGENT}


# ==============================
# Logging
# ==============================

def _build_logger(level=logging.INFO) -> logging.Logger:
    log = logging.getLogger("pipeline")
    if not log.handlers:
        log.setLevel(level)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(level)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.propagate = False
    return log

log = _build_logger()


# ==============================
# Server index helpers
# ==============================
def _remove_if_zero(path: Path) -> None:
    """Remove path if it exists and is 0 bytes. Suppress errors."""
    with contextlib.suppress(Exception):
        if path.exists() and path.stat().st_size == 0:
            path.unlink()

def _get_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def _months_from_index(html: str) -> List[str]:
    # Directories like 2023-05/
    months = set(re.findall(r'href="(20\d{2}-\d{2})/?', html, flags=re.I))
    return sorted(months)

def _estab_files_from_index(html: str) -> List[str]:
    return sorted(set(re.findall(r'href="(Estabelecimentos\d+\.zip)"', html, flags=re.I)))

def _discover_months(base_urls: Iterable[str]) -> Tuple[str, List[str]]:
    """
    Return (base_url, sorted_months) for the first base that lists months and has Estabelecimentos files.
    """
    last_err: Optional[Exception] = None
    for base in base_urls:
        try:
            idx = _get_text(base)
            months = _months_from_index(idx)
            if not months:
                continue
            # Keep only months that actually contain Estabelecimentos files
            usable: List[str] = []
            for m in months:
                sub = _get_text(f"{base}{m}/")
                if _estab_files_from_index(sub):
                    usable.append(m)
            if usable:
                return base, usable
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to list months on any base. Last error: {last_err}")


# ==============================
# Downloader (parallel ranged, fallback)
# ==============================

def _single_stream(url: str, dst: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        tmp.replace(dst)
    except Exception:
        # avoid leaving a 0-byte temp file
        _remove_if_zero(tmp)
        raise

def _download_range(url: str, start: int, end: int, out_path: Path, chunk_size: int) -> None:
    headers = dict(HEADERS)
    headers["Range"] = f"bytes={start}-{end}"
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
            if r.status_code not in (200, 206):
                r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
    except Exception:
        _remove_if_zero(out_path)
        raise

def _parallel_download(url: str, dst: Path, parts: int = 8, chunk_size: int = 8 * 1024 * 1024) -> None:
    """
    Parallel ranged download; falls back to single-stream if ranges unsupported or size unknown.
    """
    if dst.exists() and dst.stat().st_size > 0:
        log.info("Exists, skipping: %s", dst.name)
        return

    # HEAD: size + ranges
    h = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    h.raise_for_status()
    size = int(h.headers.get("Content-Length", "0"))
    ranges_ok = h.headers.get("Accept-Ranges", "").lower() == "bytes"
    host = urlparse(h.url).netloc
    log.info("Host: %s | File: %s | Size: %s | Ranges: %s",
             host, dst.name, f"{size:,}" if size else "unknown", "yes" if ranges_ok else "no")

    if (not ranges_ok) or (size == 0) or parts <= 1:
        log.info("Falling back to single-stream download.")
        _single_stream(url, dst, chunk_size)
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    seg = math.ceil(size / parts)
    ranges = [(i * seg, min(size - 1, (i + 1) * seg - 1)) for i in range(parts)]
    part_files = [tmp.with_suffix(tmp.suffix + f".{i}") for i in range(parts)]

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=parts) as ex:
        futs = []
        for i, (a, b) in enumerate(ranges):
            futs.append(ex.submit(_download_range, url, a, b, part_files[i], chunk_size))
        for f in as_completed(futs):
            f.result()
    dt = max(1e-6, time.monotonic() - t0)
    avg = size / dt if size else 0.0
    log.info("Downloaded in %.1fs, avg %.2f MB/s (%.1f Mb/s)", dt, avg / 1e6, avg * 8 / 1e6 if avg else 0.0)

    # Stitch parts
    try:
        with open(tmp, "wb") as out:
            for pf in part_files:
                with open(pf, "rb") as inp:
                    shutil.copyfileobj(inp, out)
                os.remove(pf)
        tmp.replace(dst)
    except Exception:
        _remove_if_zero(tmp)
        raise

def download_month_estabelecimentos(base: str, month: str, out_dir: Path = RAW_ROOT, parts: int = 8) -> List[Path]:
    """
    Download all Estabelecimentos*.zip for a month into data/raw/<month>/.
    """
    if not re.fullmatch(r"20\d{2}-\d{2}", month):
        raise ValueError("month must be 'YYYY-MM'")
    idx = _get_text(f"{base}{month}/")
    files = _estab_files_from_index(idx)
    if not files:
        raise RuntimeError(f"No Estabelecimentos*.zip at {base}{month}/")
    dest = out_dir / month
    dest.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    for fn in files:
        url = f"{base}{month}/{fn}"
        dst = dest / fn
        log.info("Downloading %s", fn)
        _parallel_download(url, dst, parts=max(2, min(parts, (os.cpu_count() or 4))))
        out_paths.append(dst)
    return out_paths


# ==============================
# Build monthly pairs snapshot (lazy Polars)
# ==============================

def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _list_estab_zips(month_dir: Path) -> List[Path]:
    return sorted(month_dir.glob("Estabelecimentos*.zip"))

def _extract_csv_from_zip(zip_path: Path, out_root: Path = TMP_ROOT) -> Path:
    """
    Extract first CSV/TXT from ZIP into TMP_ROOT/<YYYY-MM>/<zip-stem>/<csv>; idempotent.
    """
    month = zip_path.parent.name
    dest_dir = out_root / month / zip_path.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [zi for zi in zf.infolist() if zi.filename.lower().endswith((".csv", ".txt", ".estabele"))]
            if not members:
                raise RuntimeError(f"No CSV/TXT found inside {zip_path.name}")
            member = members[0]
            out_path = dest_dir / Path(member.filename).name
            if not out_path.exists():
                try:
                    with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                except Exception:
                    _remove_if_zero(out_path)
                    raise
            return out_path
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Bad ZIP file: {zip_path}") from e
        

FIXED_COLUMNS = ["CNPJ_BASICO", "CNPJ_ORDEM", "CNPJ_DV",
                                "IDENTIFICADOR_MATRIZ_FILIAL", 
                                "NOME_FANTASIA", 
                                "SITUACAO_CADASTRAL", "DATA_SITUACAO_CADASTRAL", "MOTIVO_SITUACAO_CADASTRAL", 
                                "NOME_CIDADE", "PAIS", 
                                "DATA_INI_ATIVIDADE", 
                                "CNAE_FISCAL_PRINCIPAL", "CNAE_FISCAL_SECUNDARIA",
                                "TIPO_LOGRADOURO", "LOGRADOURO", "NUMERO", "COMPLEMENTO", "BAIRRO", "CEP", "UF", "MUNICIPIO",
                                "DDD_1", "TELEFONE_1", "DDD_2", "TELEFONE_2", "DDD_FAX", "FAX", "CORREIO_ELETRONICO",
                                "SITUACAO_ESPECIAL", "DATA_SITUACAO_ESPECIAL"]
    
   
def _lazy_estab_csv_to_pairs(csv_path: Path) -> pl.LazyFrame:
    lf = (
        pl.scan_csv(csv_path, separator=";", encoding="utf8-lossy", has_header=False,
                    new_columns = FIXED_COLUMNS,
                    infer_schema_length=100, low_memory=True, quote_char='"', ignore_errors=True)
        .select(
            pl.col("CNPJ_BASICO").cast(pl.Utf8).alias("cnpj_basico"),
            pl.col("CNPJ_ORDEM").cast(pl.Utf8).alias("cnpj_ordem"),
            pl.col("CNPJ_DV").cast(pl.Utf8).alias("cnpj_dv"),
            pl.col("CNAE_FISCAL_PRINCIPAL").cast(pl.Utf8).alias("cnae_principal_raw"),
            pl.col("CNAE_FISCAL_SECUNDARIA").cast(pl.Utf8).alias("cnae_sec_raw"),
        )
        .with_columns(
            pl.col("cnpj_basico").str.replace_all(r"\D", "").str.zfill(8),
            pl.col("cnpj_ordem").str.replace_all(r"\D", "").str.zfill(4),
            pl.col("cnpj_dv").str.replace_all(r"\D", "").str.zfill(2),
            pl.col("cnae_principal_raw").fill_null("").str.replace_all(r"\D", "").str.zfill(7).alias("cnae_principal"),
            pl.col("cnae_sec_raw").fill_null("").cast(pl.Utf8),
        )
        .with_columns(
            pl.col("cnae_sec_raw").str.replace_all(r"[^0-9,\|\;]", "").str.replace_all(r"[|\;]", ",").alias("cnae_sec_clean"),
        )
        .with_columns(
            pl.when(pl.col("cnae_sec_clean") == "")
              .then(pl.col("cnae_principal"))
              .otherwise(pl.concat_str([pl.col("cnae_principal"), pl.lit(","), pl.col("cnae_sec_clean")]))
              .alias("cnae_all"),
        )
        .with_columns(
            pl.col("cnae_all").str.split(",")
              .list.eval(pl.element().str.replace_all(r"\D", "").str.zfill(7))
              .list.eval(pl.element().filter(pl.element().str.len_chars() == 7))
              .list.unique().alias("cnae_list"),
            pl.concat_str([pl.col("cnpj_basico"), pl.col("cnpj_ordem"), pl.col("cnpj_dv")]).alias("cnpj"),
        )
        .select("cnpj", "cnae_list")
        .explode("cnae_list")
        .rename({"cnae_list": "cnae"})
        .filter(
            (pl.col("cnpj").str.len_chars() == 14)
            & (pl.col("cnae").str.len_chars() == 7)
            & (pl.col("cnae") != "0000000")
        )
    )
    return lf

def build_months_pair(month: str,
                      *,
                      raw_root: Path = RAW_ROOT,
                      tmp_root: Path = TMP_ROOT,
                      out_root: Path = WORKING_DIR,
                      compression: str = PARQUET_COMPRESSION) -> Path:
    """
    Build unique (cnpj,cnae) snapshot for a given month (YYYY-MM) into working/<month>_pairs.parquet
    """
    if not re.fullmatch(r"20\d{2}-\d{2}", month):
        raise ValueError("month must be 'YYYY-MM' starting with '20'")
    month_dir = raw_root / month
    if not month_dir.exists():
        raise FileNotFoundError(f"Month folder not found: {month_dir}")
    zips = _list_estab_zips(month_dir)
    if not zips:
        raise RuntimeError(f"No Estabelecimentos*.zip in {month_dir}")

    lazy_parts: List[pl.LazyFrame] = []
    extracted_subdirs: List[Path] = []
    for zp in zips:
        csv_path = _extract_csv_from_zip(zp, out_root=tmp_root)
        lazy_parts.append(_lazy_estab_csv_to_pairs(csv_path))
        extracted_subdirs.append(tmp_root / month / zp.stem)

    lf = pl.concat(lazy_parts, how="vertical_relaxed").unique(subset=["cnpj", "cnae"]).sort(["cnpj", "cnae"])

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{month}_pairs.parquet"
    out_tmp = out_path.with_suffix(out_path.suffix + ".next")
    try:
        lf.sink_parquet(out_tmp, compression=compression, statistics=True)
        os.replace(out_tmp, out_path)
    except Exception:
        _remove_if_zero(out_tmp)
        raise
    # Cleanup extracted CSVs
    for sub in extracted_subdirs:
        if sub.exists():
            shutil.rmtree(sub, ignore_errors=True)

    log.info("Built pairs: %s", out_path)
    return out_path


# ==============================
# SCD bootstrap
# ==============================

def bootstrap_scd(start_month: str) -> Tuple[Path, Path]:
    """
    Initialize SCD store from <start_month>_pairs.parquet:
      - open/open_intervals.parquet (valid_to=NULL)
      - history/history_closed.parquet (empty)
    """
    pairs_path = WORKING_DIR / f"{start_month}_pairs.parquet"
    if not pairs_path.exists():
        # build if missing
        build_months_pair(start_month)

    OPEN_DIR.mkdir(parents=True, exist_ok=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)

    open_out = OPEN_DIR / "open_intervals.parquet"
    hist_out = HIST_DIR / "history_closed.parquet"

    lf_pairs = pl.scan_parquet(pairs_path)
    lf_open = (
        lf_pairs
        .with_columns(
            pl.lit(start_month).alias("valid_from_month"),
            pl.lit(None, dtype=pl.Utf8).alias("valid_to_month"),
        )
        .select("cnpj", "cnae", "valid_from_month", "valid_to_month")
        .sort(["cnpj", "cnae"])
    )
    open_tmp = open_out.with_suffix(open_out.suffix + ".next")
    try:
        lf_open.sink_parquet(open_tmp, compression=PARQUET_COMPRESSION, statistics=True)
        os.replace(open_tmp, open_out)
    except Exception:
        _remove_if_zero(open_tmp)
        raise

    # Empty history with schema
    empty_history = pl.DataFrame(
        schema={
            "cnpj": pl.Utf8,
            "cnae": pl.Utf8,
            "valid_from_month": pl.Utf8,
            "valid_to_month": pl.Utf8,
        }
    )
    hist_tmp = hist_out.with_suffix(hist_out.suffix + ".next")
    try:
        empty_history.write_parquet(hist_tmp, compression=PARQUET_COMPRESSION)
        os.replace(hist_tmp, hist_out)
    except Exception:
        _remove_if_zero(hist_tmp)
        raise

    # Sanity: counts match
    pairs_cnt = lf_pairs.select(pl.len()).collect(engine = "streaming").item()
    open_cnt  = pl.scan_parquet(open_out).select(pl.len()).collect(engine = "streaming").item()
    if pairs_cnt != open_cnt:
        raise RuntimeError(f"Bootstrap count mismatch: pairs={pairs_cnt} open={open_cnt}")

    log.info("Bootstrap complete. open=%s  history=%s", open_out, hist_out)
    return open_out, hist_out

# ----------------------------
# SCD update for one month
# ----------------------------
def _is_good_parquet(path: Path) -> bool:
    try:
        if (not path.exists()) or path.stat().st_size == 0:
            return False
        # small probe; fail fast if corrupted
        pl.scan_parquet(path).select(pl.len()).collect(engine = "streaming")
        return True
    except Exception:
        return False

def scd_update(month: str, prev_month: str) -> None:
    prev_pairs = WORKING_DIR / f"{prev_month}_pairs.parquet"
    curr_pairs = WORKING_DIR / f"{month}_pairs.parquet"
    if not prev_pairs.exists() or not curr_pairs.exists():
        raise FileNotFoundError("Build both <prev> and <month> pairs before scd_update")

    S_prev = pl.scan_parquet(prev_pairs)  # (cnpj, cnae)
    S_curr = pl.scan_parquet(curr_pairs)

    # Δ sets
    added   = S_curr.join(S_prev, on=["cnpj", "cnae"], how="anti")  # present now, absent before
    removed = S_prev.join(S_curr, on=["cnpj", "cnae"], how="anti")  # present before, absent now

    open_path = OPEN_DIR / "open_intervals.parquet"

    # Guard: verify current open is readable; if not, reconstruct from prev snapshot
    if not _is_good_parquet(open_path):
        # reconstruct open from S_prev (valid_to=NULL)
        OPEN_DIR.mkdir(parents=True, exist_ok=True)
        recon = (
            S_prev
            .with_columns(
                pl.lit(prev_month).alias("valid_from_month"),
                pl.lit(None, dtype=pl.Utf8).alias("valid_to_month"),
            )
            .select("cnpj", "cnae", "valid_from_month", "valid_to_month")
            .sort(["cnpj", "cnae"])
        )
        tmp_recon = open_path.with_suffix(open_path.suffix + ".rebuild.tmp")
        try:
            recon.sink_parquet(tmp_recon, compression=PARQUET_COMPRESSION, statistics=True)
            os.replace(tmp_recon, open_path)
        except Exception:
            _remove_if_zero(tmp_recon)
            raise

    open_lf = pl.scan_parquet(open_path)

    # Close intervals for removed pairs (valid_to = prev_month)
    to_close = (
        removed.join(
            open_lf.select("cnpj", "cnae", "valid_from_month"),
            on=["cnpj", "cnae"],
            how="inner"
        )
        .with_columns(pl.lit(prev_month).alias("valid_to_month"))
        .select("cnpj", "cnae", "valid_from_month", "valid_to_month")
        .sort(["cnpj", "cnae"])
    )

    # Write a new history *part* (no appends to a single file)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    hist_part = HIST_DIR / f"closed_until_{prev_month}.parquet"
    n_close = to_close.select(pl.len()).collect(engine="streaming").item()
    if n_close > 0:
        df_close = to_close.collect(engine="streaming")
        hist_tmp = hist_part.with_suffix(hist_part.suffix + ".next")
        try:
            df_close.write_parquet(hist_tmp, compression=PARQUET_COMPRESSION)
            os.replace(hist_tmp, hist_part)
        except Exception:
            _remove_if_zero(hist_tmp)
            raise
    else:
        # skip writing an empty history file entirely
        pass

    
    # Update open: remove closed keys, add new opens
    still_open = open_lf.join(removed, on=["cnpj", "cnae"], how="anti")
    new_open = (
        added
        .with_columns(
            pl.lit(month).alias("valid_from_month"),
            pl.lit(None, dtype=pl.Utf8).alias("valid_to_month"),
        )
        .select("cnpj", "cnae", "valid_from_month", "valid_to_month")
    )
    updated_open = (
        pl.concat([still_open, new_open], how="vertical_relaxed")
        .unique(subset=["cnpj", "cnae"])
        .sort(["cnpj", "cnae"])
    )

    # >>> sink to TEMP, then atomic replace <<<
    OPEN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df_open = updated_open.collect(engine="streaming")
        open_tmp = open_path.with_suffix(open_path.suffix + ".next")
        try:
            df_open.write_parquet(open_tmp, compression=PARQUET_COMPRESSION)
            os.replace(open_tmp, open_path)
        except Exception:
            _remove_if_zero(open_tmp)
            raise
    except Exception as e:
        # ensure we don't leave a 0-byte temp
        with contextlib.suppress(Exception):
            if open_tmp.exists():
                _remove_if_zero(open_tmp)
        raise 

    # Sanity: keys(open) must equal keys(S_curr)
    cnt_open = pl.scan_parquet(open_path).select(pl.len()).collect(engine = "streaming").item()
    cnt_curr = S_curr.select(pl.len()).collect(engine = "streaming").item()
    if cnt_open != cnt_curr:
        raise RuntimeError(f"SCD check failed for {month}: open={cnt_open} != current={cnt_curr}")


# ==============================
# Pipeline
# ==============================

def run_pipeline(start_month: str,
                 *,
                 end_month: str | None = None,
                 download_parts: int = 8,
                 log_level: int = logging.INFO) -> None:
    """
    1) Discover months and download all ≥ start_month (≤ end_month if set)
    2) Build pairs & bootstrap at start_month
    3) For each subsequent month, build pairs and run scd_update
    """
    _build_logger(log_level)

    base, months = _discover_months(BASE_URLS)
    months = [m for m in months if m >= start_month and (m <= end_month if end_month else True)]
    if not months:
        raise RuntimeError("No months in selected range on server.")

    # Download all months in range
    for m in months:
        download_month_estabelecimentos(base, m, out_dir=RAW_ROOT, parts=download_parts)

    # Build start month pairs + bootstrap
    build_months_pair(start_month)
    bootstrap_scd(start_month)

    # Advance SCD for later months
    for prev, curr in zip(months, months[1:]):
        build_months_pair(curr)
        scd_update(curr, prev)



# ==============================
# CLI
# ==============================

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download Estabelecimentos and bootstrap SCD DDBB")
    ap.add_argument("--start", required=True, help="Start month YYYY-MM (e.g., 2023-05)")
    ap.add_argument("--end", help="End month YYYY-MM (defaults to latest available on server)")
    ap.add_argument("--parts", type=int, default=8, help="Parallel parts per file (default 8)")
    ap.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.start, end_month=args.end, download_parts=args.parts,
                 log_level=getattr(logging, args.log.upper()))
