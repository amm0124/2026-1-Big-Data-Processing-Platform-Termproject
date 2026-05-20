"""PoC: 선수/볼 궤적별 선형 보간 복원.

데이터 규약은 /README.md를 따름:
  - 입력: processed/{XX}match_tracking_data.npz
  - NPZ는 다음 6개 키를 갖는 dict 형태:
        fh_home, fh_away, fh_ball, sh_home, sh_away, sh_ball
  - 각 배열 shape = (T, N*2). N은 엔티티 수, 컬럼 순서는
        [e0_x, e0_y, e1_x, e1_y, ...]. 샘플링 주기는 25 fps (0.04 s).

처리 파이프라인 (term-project 논문):
  1. 0.04 s NPZ 데이터를 ground truth P로 간주
  2. 짝수 인덱스 프레임(0, 2, 4, ...)만 남겨 0.08 s 표현 P_low 생성
  3. 제거된 홀수 인덱스 프레임은 양쪽 이웃의 중점으로 복원
        p_hat_t = 0.5 * (p_{t-1} + p_{t+1})
  4. GT와 복원 궤적 사이의 RMSE / discrete Frechet distance 계산
  5. 궤적 비교 + 샘플별 오차 막대그래프 시각화
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc

# src/visualization.py의 스타일 컨벤션을 그대로 차용
# (floodlight 패키지가 설치돼 있으면 그쪽의 pitch.plot()이 우선)
plt.style.use("ggplot")
COL_FACE = "lightgrey"

# floodlight가 설치돼 있으면 import — 없으면 자체 draw_pitch로 대체
try:
    from floodlight.io.dfl import read_pitch_from_mat_info_xml  # noqa: F401
    HAS_FLOODLIGHT = True
except ImportError:
    HAS_FLOODLIGHT = False

# ── 경로 설정 ───────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(HERE, "..", "..", "processed")  # NPZ 입력 폴더
FIG_DIR = os.path.join(HERE, "figures")                      # 시각화 출력
RESULTS_PATH = os.path.join(HERE, "results.txt")             # 수치 결과 출력
os.makedirs(FIG_DIR, exist_ok=True)

# README가 명시한 6개 키 — 가능한 모든 키를 순회
NPZ_KEYS = ["fh_home", "fh_away", "fh_ball", "sh_home", "sh_away", "sh_ball"]

# ── 피치 규격 (FIFA 표준, DFL 매치정보 XML 기준 105×68 m) ────────────────────
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_XLIM = (-PITCH_LENGTH / 2, PITCH_LENGTH / 2)   # = (-52.5, 52.5)
PITCH_YLIM = (-PITCH_WIDTH / 2, PITCH_WIDTH / 2)     # = (-34.0, 34.0)

# ── 실험 하이퍼파라미터 ──────────────────────────────────────────────────────
SEG_LEN = 500       # 한 궤적 길이: 500 프레임 = 25 fps 기준 20초
N_SAMPLES = 10      # 평가할 샘플 개수
RNG_SEED = 42       # 프로젝트 전반에서 사용하는 split seed와 동일


def load_match_npz(npz_path):
    """README 포맷의 NPZ를 {key: ndarray} dict로 로드."""
    # np.load는 lazy file handle이므로 with 블록으로 닫아줌
    with np.load(npz_path, allow_pickle=True) as d:
        # NPZ에 실제로 존재하는 키만 가져옴 (안전망)
        return {k: d[k] for k in NPZ_KEYS if k in d.files}


def find_valid_segments(entity_xy, seg_len):
    """한 엔티티(선수/볼) 좌표에서 NaN이 없는 연속 구간을 찾음.

    반환: [(start, end), ...]. end는 exclusive. seg_len 이상인 구간만 유지.
    NaN은 출전하지 않았거나 데이터가 끊긴 프레임을 의미.
    """
    # x 또는 y가 하나라도 NaN인 프레임은 유효하지 않음
    valid = ~np.isnan(entity_xy).any(axis=1)
    segments = []
    run_start = None  # 현재 진행 중인 valid run의 시작 인덱스
    for i, v in enumerate(valid):
        if v and run_start is None:
            # valid run 시작
            run_start = i
        elif not v and run_start is not None:
            # valid run 종료 — 길이가 충분하면 저장
            if i - run_start >= seg_len:
                segments.append((run_start, i))
            run_start = None
    # 배열 끝까지 valid가 이어진 경우의 처리
    if run_start is not None and len(valid) - run_start >= seg_len:
        segments.append((run_start, len(valid)))
    return segments


def enumerate_entities(match_dict):
    """모든 (key, entity_idx, xy_array) 조합을 yield.

    예) fh_home 배열 shape이 (T, 40)이면 entity 0~19에 대해 20번 yield.
    """
    for key, arr in match_dict.items():
        # 컬럼은 (x, y) 페어이므로 엔티티 수 = 컬럼 / 2
        n_entities = arr.shape[1] // 2
        for e in range(n_entities):
            # 해당 엔티티의 (x, y) 두 컬럼만 슬라이스
            yield key, e, arr[:, 2 * e : 2 * e + 2]


def collect_samples(match_dict, n_samples, seg_len, rng):
    """6개 키에 골고루 분산되도록 n_samples개 궤적을 추출.

    1단계: 키마다 1개씩 우선 뽑아 다양성 확보 (home/away/ball, 1·2하프 모두)
    2단계: 남은 슬롯을 임의로 채움
    """
    # 모든 (key, entity, valid_start, valid_end) 후보를 수집
    candidates = []
    for key, e_idx, xy in enumerate_entities(match_dict):
        for s, end in find_valid_segments(xy, seg_len):
            candidates.append((key, e_idx, s, end))
    if not candidates:
        return []
    # seed 기반으로 셔플 (재현 가능)
    rng.shuffle(candidates)

    samples = []
    seen_keys = set()

    # 1단계: 키별로 1개씩 — 결과의 대표성을 높이기 위함
    for key, e_idx, s, end in candidates:
        if key in seen_keys:
            continue
        # valid 구간 내에서 무작위로 시작점 선택
        start = int(rng.integers(s, end - seg_len + 1))
        xy = match_dict[key][start : start + seg_len, 2 * e_idx : 2 * e_idx + 2].copy()
        samples.append({"key": key, "entity": e_idx, "start": start, "gt": xy})
        seen_keys.add(key)
        if len(samples) == n_samples:
            return samples

    # 2단계: 키 제약 없이 나머지 슬롯 채우기
    for key, e_idx, s, end in candidates:
        if len(samples) == n_samples:
            break
        start = int(rng.integers(s, end - seg_len + 1))
        xy = match_dict[key][start : start + seg_len, 2 * e_idx : 2 * e_idx + 2].copy()
        samples.append({"key": key, "entity": e_idx, "start": start, "gt": xy})
    return samples


def downsample_and_reconstruct(gt):
    """0.04 s GT를 0.08 s로 다운샘플 후 선형 보간으로 다시 복원.

    Retained (저장): 짝수 인덱스 (0, 2, 4, ...) → 0.08 s 표현
    Missing  (제거): 홀수 인덱스 (1, 3, 5, ...) → 중점 보간으로 재구성

    수식 (논문): p_hat_t = 0.5 * (p_{t-1} + p_{t+1})
    """
    # 원본에 그대로 남는 프레임 인덱스
    retained_idx = np.arange(0, len(gt), 2)
    # 복원 대상 프레임 인덱스 (양쪽 이웃이 모두 있어야 하므로 마지막은 제외)
    missing_idx = np.arange(1, len(gt) - 1, 2)

    recon = gt.copy()
    # 양쪽 이웃 평균 — numpy 벡터 연산으로 한 번에 처리
    recon[missing_idx] = 0.5 * (gt[missing_idx - 1] + gt[missing_idx + 1])

    # 궤적이 홀수 인덱스에서 끝나면 오른쪽 이웃이 없어 경계 처리 필요
    # 가장 단순한 hold-last 정책: 직전 retained 값으로 채움
    if (len(gt) - 1) % 2 == 1:
        recon[-1] = gt[-2]
    return recon, retained_idx, missing_idx


def rmse(a, b):
    """프레임별 L2 오차의 제곱평균제곱근. 단위는 m (피치 좌표계)."""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def discrete_frechet(p, q):
    """Eiter-Mannila DP 기반 discrete Frechet distance.

    p: [m, 2], q: [n, 2]. 두 궤적 사이의 '리쉬 길이' — 단조 결합 중
    최댓값을 최소화한 값. RMSE가 평균 오차라면 Frechet은 worst-case 형태.
    """
    m, n = len(p), len(q)
    # 모든 (i, j) 쌍의 유클리드 거리 — 한 번에 계산 (메모리 m*n)
    d = np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(axis=-1))
    ca = np.empty_like(d)

    # 경계 조건: 첫 행/첫 열은 누적 max로 채움
    ca[0, 0] = d[0, 0]
    for i in range(1, m):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, n):
        ca[0, j] = max(ca[0, j - 1], d[0, j])

    # DP 본체: 세 방향(↑, ↖, ←) 중 최소값을 받아 현재 거리와 max
    for i in range(1, m):
        for j in range(1, n):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d[i, j])
    return float(ca[-1, -1])


def evaluate(samples):
    """각 샘플마다 복원 + 두 가지 metric 계산."""
    results = []
    for s in samples:
        gt = s["gt"]
        recon, ret_idx, miss_idx = downsample_and_reconstruct(gt)
        results.append(
            {
                **s,
                "recon": recon,
                "retained_idx": ret_idx,
                "missing_idx": miss_idx,
                # 복원된 프레임에서만 계산 — baseline의 '진짜' 복원 정확도
                "rmse_missing": rmse(gt[miss_idx], recon[miss_idx]),
                # 전체 궤적 RMSE — retained 프레임은 오차 0이라 평균이 낮아짐
                "rmse_full": rmse(gt, recon),
                # 궤적 형태 유사도 — 큰 국소 편차에 민감
                "frechet": discrete_frechet(gt, recon),
            }
        )
    return results


def sample_label(r):
    """그래프 x축 라벨용 짧은 식별자 — 예: 'fh_home/e3 @1234'."""
    return f"{r['key']}/e{r['entity']}\n@{r['start']}"


def draw_pitch(ax, line_color="white", line_width=1.2):
    """src/visualization.py의 floodlight `pitch.plot(ax=ax)` 패턴 대체.

    floodlight 패키지가 설치돼 있지 않은 환경에서도 동일한 외관(중앙 원점
    105x68 m FIFA 피치)을 그릴 수 있도록 직접 라인을 그림.
    visualization.py의 COL_FACE("lightgrey")를 그대로 사용해 톤을 일치시킴.
    """
    ax.set_facecolor(COL_FACE)
    ax.set_xlim(-55, 55)   # visualization.py와 동일한 여유 마진
    ax.set_ylim(-37, 37)
    ax.set_aspect("equal")

    # 외곽선
    ax.add_patch(Rectangle(
        (-PITCH_LENGTH / 2, -PITCH_WIDTH / 2),
        PITCH_LENGTH, PITCH_WIDTH,
        fill=False, edgecolor=line_color, linewidth=line_width,
    ))

    # 하프웨이 라인
    ax.plot([0, 0], [-PITCH_WIDTH / 2, PITCH_WIDTH / 2],
            color=line_color, linewidth=line_width)

    # 센터 서클 + 센터 스팟
    ax.add_patch(Circle((0, 0), 9.15,
                        fill=False, edgecolor=line_color, linewidth=line_width))
    ax.add_patch(Circle((0, 0), 0.3, color=line_color))

    # 페널티 박스 / 골 박스 / 페널티 스팟 / 페널티 아크 (양쪽)
    for side in (-1, +1):
        # 페널티 박스 (16.5m × 40.32m)
        ax.add_patch(Rectangle(
            (side * (PITCH_LENGTH / 2 - 16.5) - (16.5 if side > 0 else 0), -20.16),
            16.5, 40.32,
            fill=False, edgecolor=line_color, linewidth=line_width,
        ))
        # 골 박스 (5.5m × 18.32m)
        ax.add_patch(Rectangle(
            (side * (PITCH_LENGTH / 2 - 5.5) - (5.5 if side > 0 else 0), -9.16),
            5.5, 18.32,
            fill=False, edgecolor=line_color, linewidth=line_width,
        ))
        # 페널티 스팟 (골라인에서 11 m)
        ax.add_patch(Circle((side * (PITCH_LENGTH / 2 - 11), 0), 0.3,
                            color=line_color))
        # 페널티 아크 — 페널티 스팟 기준 9.15 m, 박스 바깥쪽만
        theta1, theta2 = (130, 230) if side > 0 else (-50, 50)
        ax.add_patch(Arc(
            (side * (PITCH_LENGTH / 2 - 11), 0), 2 * 9.15, 2 * 9.15,
            angle=0, theta1=theta1, theta2=theta2,
            color=line_color, linewidth=line_width,
        ))

    ax.tick_params(labelsize=7)


def plot_trajectories(results, out_path):
    """샘플별로 피치 위에 GT 라인 + retained 점 + 복원 점을 그림.

    src/visualization.py가 floodlight의 pitch.plot(ax=ax) + 궤적 오버레이
    조합으로 그림을 만드는 흐름을 그대로 따른다.
    """
    n = len(results)
    cols = 2
    rows = (n + cols - 1) // cols  # 올림
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes).ravel()

    for ax, r in zip(axes, results):
        # 피치 배경 먼저 — 궤적 아래에 깔리도록
        draw_pitch(ax)

        gt = r["gt"]
        recon = r["recon"]
        miss = r["missing_idx"]

        # GT: 파란 라인 (전체 0.04 s 궤적)
        ax.plot(gt[:, 0], gt[:, 1], "-", color="tab:blue", lw=1.2, alpha=0.7,
                label="GT (0.04 s)")
        # Retained: 초록 점 (실제로 저장되는 0.08 s 샘플)
        ax.scatter(gt[r["retained_idx"], 0], gt[r["retained_idx"], 1],
                   s=10, color="tab:green", label="retained (0.08 s)", zorder=3)
        # 복원값: 빨간 ×
        ax.scatter(recon[miss, 0], recon[miss, 1],
                   s=14, color="tab:red", marker="x", label="reconstructed", zorder=4)

        ax.set_title(
            f"{r['key']}  entity={r['entity']}  start={r['start']}\n"
            f"RMSE={r['rmse_missing']:.3f} m   Frechet={r['frechet']:.3f} m",
            fontsize=9,
        )

    # 범례는 첫 번째 subplot에만 — 중복 방지
    axes[0].legend(fontsize=7, loc="upper left")
    # 샘플 수가 홀수면 마지막 칸이 비므로 숨김
    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle("Linear-interpolation reconstruction vs ground truth", fontsize=12)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_error_summary(results, out_path):
    """샘플별 RMSE / Frechet 막대그래프 — outlier 식별용."""
    rmses = [r["rmse_missing"] for r in results]
    frechets = [r["frechet"] for r in results]
    idx = np.arange(len(results))
    labels = [sample_label(r) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # 왼쪽: RMSE 막대 + 평균선
    axes[0].bar(idx, rmses, color="tab:red", alpha=0.8)
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("RMSE on reconstructed frames (m)")
    axes[0].set_title(f"Per-sample RMSE  (mean={np.mean(rmses):.4f} m)")
    axes[0].axhline(np.mean(rmses), color="k", lw=0.8, ls="--")

    # 오른쪽: Frechet 막대 + 평균선
    axes[1].bar(idx, frechets, color="tab:purple", alpha=0.8)
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("Discrete Frechet distance (m)")
    axes[1].set_title(f"Per-sample Frechet  (mean={np.mean(frechets):.4f} m)")
    axes[1].axhline(np.mean(frechets), color="k", lw=0.8, ls="--")

    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def format_results_table(results):
    """수치 결과를 보기 쉬운 텍스트 표로 정렬."""
    lines = []
    header = f"{'idx':>3} {'key':>8} {'entity':>6} {'start':>7} {'RMSE_miss':>10} {'RMSE_full':>10} {'Frechet':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, r in enumerate(results):
        lines.append(
            f"{i:>3} {r['key']:>8} {r['entity']:>6} {r['start']:>7} "
            f"{r['rmse_missing']:>10.4f} {r['rmse_full']:>10.4f} {r['frechet']:>10.4f}"
        )
    lines.append("")
    # 요약 통계 — 평균값으로 baseline 전반 성능 가늠
    lines.append(f"mean RMSE (reconstructed frames only) : {np.mean([r['rmse_missing'] for r in results]):.4f} m")
    lines.append(f"mean RMSE (full trajectory)           : {np.mean([r['rmse_full'] for r in results]):.4f} m")
    lines.append(f"mean discrete Frechet distance        : {np.mean([r['frechet'] for r in results]):.4f} m")
    return "\n".join(lines)


def main():
    # processed 폴더의 첫 번째 매치 파일을 사용 (정렬 후 가장 앞)
    npz_files = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith(".npz"))
    if not npz_files:
        print(f"no NPZ files in {PROCESSED_DIR}", file=sys.stderr)
        sys.exit(1)
    match_path = os.path.join(PROCESSED_DIR, npz_files[0])
    print(f"loading {match_path}")

    # 매치 로드 + 각 키별 shape 출력 (sanity check)
    match = load_match_npz(match_path)
    for k, v in match.items():
        print(f"  {k}: shape={v.shape}")

    # 시드 고정 RNG로 샘플 추출 — 재현 가능
    rng = np.random.default_rng(RNG_SEED)
    samples = collect_samples(match, N_SAMPLES, SEG_LEN, rng)
    print(f"\ncollected {len(samples)} trajectories "
          f"(seg_len={SEG_LEN} frames = {SEG_LEN * 0.04:.1f} s)")

    # 다운샘플 → 복원 → metric 계산
    results = evaluate(samples)

    # 콘솔 출력 + 파일 저장
    table = format_results_table(results)
    print()
    print(table)

    with open(RESULTS_PATH, "w") as f:
        f.write(f"source: {os.path.relpath(match_path, HERE)}\n")
        f.write(f"seg_len={SEG_LEN} frames, n_samples={N_SAMPLES}, seed={RNG_SEED}\n\n")
        f.write(table + "\n")

    # 시각화 — figures/ 폴더에 PNG 저장
    traj_path = os.path.join(FIG_DIR, "trajectories.png")
    err_path = os.path.join(FIG_DIR, "error_summary.png")
    plot_trajectories(results, traj_path)
    plot_error_summary(results, err_path)
    print()
    print(f"saved: {RESULTS_PATH}")
    print(f"saved: {traj_path}")
    print(f"saved: {err_path}")


if __name__ == "__main__":
    main()
