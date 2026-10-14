# grid-cells-torch 재정비 + A2C 전환 + goal-finding 재현 통합 계획

## 0. 배경 (Context)

**지금까지의 결과.** A3C로 1e9 스텝을 완주했지만(`checkpoint_step1000000092_final.pt`)
평균 reward는 처음부터 끝까지 0.93점(골 발견율 7.7%)으로 평탄했습니다. 논문 Fig. 2f는
약 289점입니다. 5개 에피소드 영상에서도 5/5 모두 "벽까지 직진한 뒤 정지"였습니다.
코드를 읽어 actor-critic 경로의 버그 A(bootstrap에 V(s_t)를 재사용)와 버그 B(직전
reward 입력이 항상 0)를 확인했습니다.

**이번 요청으로 바뀐 것.**
1. repo 구조를 `jn12-29/grid-cells-torch`와 `inbox/restructuring-guide.md`를 참고해
   재편합니다(라이브러리 패키지화, 루트 CLI 진입점, `run_scripts.sh`, `docs/`).
2. 학습 방식을 A3C(CPU, Hogwild)에서 A2C(GPU 배치, 동기식)로 바꿉니다.
3. 데이터 생성을 CSV를 거치지 않고 시뮬레이터에서 바로 `.npz` shard로 쓰도록 바꿉니다.
4. 랩실 계정 전환과 인수인계 방식을 정합니다.
5. "매 세션 논문·figure·코드 전부 읽기" 규칙을 필요할 때만 읽는 구조로 바꿉니다.
6. 위 변경에 맞게 테스트 계획을 다시 짭니다.

**이번 조사에서 새로 찾은 것.** 논문 Methods를 다시 대조하다가 버그 A/B 외에 논문과
어긋나는 지점을 더 찾았습니다. 아래 §6의 C~G입니다.

---

## 1. 운영 원칙 (기존 유지 + 보강)

1. **게이트 없는 장기 실행 금지.** 중간 합격선을 미리 적어두고, 미달하면 멈추고 디버깅합니다.
2. **손으로 확인한 것은 전부 테스트로 고정합니다.**
3. **한 번에 하나만 바꿉니다.** 구조 재편은 동작을 바꾸지 않는 이동만 하고, 동작 변경은
   golden test로 "안 바뀌었음"을 먼저 증명한 다음에 합니다.
4. **빨간불 먼저.** 버그를 잡는 테스트는 수정 전 코드에서 실패하는 것을 먼저 확인합니다.
5. **논문과의 차이는 문서화합니다.** 논문이 명시한 것은 논문을 따르고, 논문이 침묵하는 것은
   표준 A2C 관행을 따릅니다. 어느 쪽이든 `docs/paper/deviations.md`에 기록합니다.

---

## 2. 계정 전환과 인수인계 — 의견과 절차

### 2.1 CLI에서 랩실 계정을 쓸 수 있는가 (공식 문서로 확인함)

가능합니다. 공식 문서(code.claude.com/docs/en/authentication)에 Team/Enterprise에
대해 *"log in with the claude.ai account your team admin invited you to"*라고 되어
있습니다. 전환 방법은 두 가지입니다.

- **방법 A — `/logout` → `/login`** (같은 `~/.claude` 사용): 로그인 정보
  (`~/.claude/.credentials.json`)만 바뀝니다. 같은 이메일이 개인 계정과 랩실 조직에
  모두 속해 있으면 브라우저 로그인 화면에서 조직을 고르게 됩니다. 전환 후 `/status`의
  Organization 행으로 확인합니다.
- **방법 B — `CLAUDE_CONFIG_DIR=~/.claude-lab claude`** (두 계정을 동시에 유지): 문서에
  따르면 *"Each directory has its own settings, session history, and claude.ai login"*.
  디렉터리가 통째로 갈라집니다.

### 2.2 계정이 달라서 생기는 문제

| 항목 | 방법 A (재로그인) | 방법 B (별도 config dir) |
|---|---|---|
| workspace `CLAUDE.md`, repo 파일 | 그대로 (디스크 파일) | 그대로 |
| `~/.claude/CLAUDE.md` (사용자 전역 지시) | 그대로 | **없음** → 복사하거나 symlink 필요 |
| auto memory (`~/.claude/projects/.../memory/`) | 그대로 ("machine-local") | **분리됨** → `autoMemoryDirectory`로 같은 곳을 가리키게 해야 함 |
| 세션 기록 / `/resume` | 그대로 (로컬 jsonl) | **분리됨** |
| plan 파일 (`~/.claude/plans/`) | 그대로 | 분리됨 |
| claude.ai 커넥터 (Claude Docs, Drive, Notion) | **계정에 묶임** → 랩실 계정에서 다시 연결 | 동일 |
| 개인 계정으로 publish한 Artifact | 링크는 유지되지만 랩실 계정 세션에서는 수정 불가 | 동일 |
| Remote Control, cloud 세션, `/schedule` 루틴 | 계정별 | 계정별 |
| 사용량 한도 | 랩실 seat 기준 | 동일 |
| 조직 정책 | 랩실 조직의 server-managed settings가 내 세션에 적용될 수 있음(모델·기능 제한 등), 대화 데이터는 조직 정책(보존·관리자 가시성)을 따름 | 동일 |

→ **방법 A를 권장합니다.** 로컬에 쌓인 것(CLAUDE.md, memory, plan, 세션)이 전부 그대로
이어지고, 바뀌는 것은 계정에 묶인 원격 기능뿐입니다. 이 프로젝트는 원격 기능을 거의
쓰지 않습니다.

### 2.3 결정: 계정 전환은 보류 (2026-10-08)

사용자는 지금 SSH 없이 Remote Control로만 접속하고 있습니다. `/login`은 이 컴퓨터에서
브라우저 OAuth를 거쳐야 하고, Remote Control 세션도 계정에 묶여 있어서 원격에서 전환하면
접속이 끊길 위험이 있습니다. 그래서 **당분간 개인 계정으로 진행하고, 전환은 SSH나 직접
접속이 가능해질 때** 합니다. 전환할 때 아래 절차를 그대로 씁니다. `docs/plan/` +
`STATUS.md`는 계정과 무관하게 지금부터 유지합니다(compact와 세션 교체 대비).

### 2.4 (참고) 언제 전환할지 — 의견

원래 생각하신 순서는 "개인 계정으로 테스트까지 끝내고, 학습할 때만 랩실 계정"이었습니다.
저는 **반대로, 계획 승인 직후 바로 전환하는 쪽**을 권합니다. 이유는 다음과 같습니다.

- **토큰을 많이 쓰는 건 학습이 아니라 개발입니다.** 학습은 Python 프로세스가 혼자 며칠
  도는 일이라 Claude 토큰을 거의 쓰지 않습니다(가끔 진행 상황을 확인할 때만 씁니다).
  반면 구조 재편, 테스트 작성, A2C 구현, 디버깅은 코드를 많이 읽고 쓰므로 토큰을 대량으로
  씁니다. 랩실 계정의 넉넉한 한도가 필요한 쪽은 바로 이 개발 구간입니다.
- 방법 A라면 로컬 상태가 전부 이어지므로 "개인 계정에서 정리하고 넘어가야 할 것"이
  사실상 없습니다.

**인수인계는 프롬프트보다 repo 안의 문서로 하는 편이 낫습니다.** 프롬프트는 한 번 쓰고
사라지지만, `docs/plan/` 문서는 compact 이후에도, 다음 세션에서도, 다른 사람(JW 브랜치
협업자 포함)도 다시 읽을 수 있습니다. 절차는 다음과 같습니다.

1. (개인 계정, 계획 승인 직후) 이 plan을 `grid-cells-torch/docs/plan/2026-10-goal-finding.md`로
   복사하고, 진행 상태 체크리스트(`docs/plan/STATUS.md`: Phase별 완료 여부, 마지막 결정,
   다음 할 일)를 만듭니다. 이것이 인수인계 문서입니다.
   - **plan을 repo로 옮기는 이유**: 지금 이 plan은 `~/.claude/plans/tranquil-dazzling-llama.md`,
     즉 Claude Code의 내부 폴더에 있습니다. 이름도 무작위이고 git으로 관리되지 않으며,
     다른 계정, 다른 컴퓨터, 협업자에게는 보이지 않습니다. repo의 `docs/plan/`에 두면
     커밋되고, 어느 세션이든 경로 하나로 찾을 수 있습니다(가이드의 `docs/plan/` 용도와 같음).
   - **`STATUS.md`란**: plan은 "무엇을 어떤 순서로 할지"(거의 바뀌지 않음)이고, STATUS는
     "지금 어디까지 했는지"(매 작업마다 갱신)입니다. 예시는 다음과 같습니다.
     ```markdown
     # Status — goal-finding plan
     Updated: 2026-10-09
     ## Phases
     - [x] 0 handoff + account switch
     - [x] 1 baseline frozen (docs/plan/baseline-a3c.md)
     - [ ] 2 context restructure — spec-rl-agent.md drafted, verification pending
     ## Decisions made
     - 2026-10-08: A2C on GPU; A3C kept frozen as rl_a3c.py
     ## Next
     - run note-vs-PDF verification agent on spec-rl-agent.md
     ## Open questions
     - Table 2 unroll length (100 vs 4?) — check image p.24
     ```
     세션이 끊기거나 compact되거나 계정이 바뀌어도, 새 세션은 이 파일 하나만 읽으면 바로
     이어서 작업할 수 있습니다. 각 Phase가 끝날 때마다 갱신하고 함께 커밋합니다.
2. `/logout` → `/login`(랩실 조직 선택) → `/status`로 확인합니다.
3. 랩실 계정에서는 **새 세션**을 엽니다. 이 거대한 세션을 `/resume`해도 되지만, 전체
   기록을 다시 읽느라 토큰이 크게 들어갑니다. 첫 프롬프트는 이렇게 씁니다:
   ```
   docs/plan/2026-10-goal-finding.md 와 docs/plan/STATUS.md 를 읽고,
   STATUS.md 의 "다음 할 일"부터 이어서 진행해줘.
   ```
4. Claude Docs 등 커넥터가 필요하면 랩실 계정에서 다시 연결합니다(현재 계획에는 필요 없음).

---

## 3. 컨텍스트 전략 — "매번 전부 읽기" 폐지와 `docs/paper/` 신설

### 3.1 의견

말씀하신 대로 비효율적입니다. 대략 계산하면 본문 PDF 텍스트 약 88k자(~22k 토큰),
Supplementary 약 46k자(~12k 토큰), figure 14장(이미지라 장당 수천 토큰), 코드 수천 줄입니다.
**세션마다, 그리고 compact할 때마다** 합쳐서 10만 토큰 안팎이 들어갑니다. 공식 문서도
"project-root CLAUDE.md survives compaction: after /compact, Claude re-reads it"라고 하므로,
현재 규칙은 compact할 때마다 전체 재독을 다시 촉발합니다. 그렇게 컨텍스트가 차면 정작
작업 내용이 더 빨리 밀려납니다.

다만 이 규칙에는 실제 근거가 있었습니다. figure를 안 봐서 cue를 공중에 띄우는 실수를
했습니다. 따라서 **"안 읽어도 된다"가 아니라 "필요한 부분을, 필요한 때, 정확히 찾아
읽게 한다"**로 바꿔야 합니다.

### 3.2 새 구조

```
grid-cells-torch/
├── CLAUDE.md                     # 짧게 (<150줄). 라우팅 표 + 핵심 규칙만
├── .claude/rules/
│   ├── rl-agent.md               # paths: grid_cells/rl_*.py, grid_cells/actor_critic.py, levels/**
│   └── supervised.md             # paths: grid_cells/model.py, grid_cells/training.py, grid_cells/data_generation.py
└── docs/
    ├── maintenance.md            # 패키지 경계, 진입점, 규칙 (가이드 7단계)
    ├── paper/
    │   ├── spec-supervised.md    # Table 1 + 아키텍처 + loss. 모든 항목에 (PDF p.N / Methods 절) 출처
    │   ├── spec-rl-agent.md      # RL Methods + Table 2 + Ext. Data Fig. 5. 출처 포함
    │   ├── figures.md            # figure별 "이미지에서만 보이는 사실" + 언제 원본 이미지를 열어야 하는지
    │   ├── deviations.md         # 논문과 다른 점 전부 (README의 해당 절을 이리로 옮기고 README는 링크)
    │   ├── figures/*.png         # supplements/figures 에서 이동
    │   ├── README.md             # 출처 명시 (§3.3)
    │   ├── *.pdf                 # supplements/논문 에서 이동 (기본 gitignore — §3.3)
    │   └── text/main.txt, supp.txt   # pypdf 추출본 캐시 (gitignore). grep용
    ├── plan/                     # 구현 계획, STATUS.md
    └── goals/                    # 기능 명세
```

- **라우팅 표**(CLAUDE.md): "이 작업이면 이것을 읽어라" 형식입니다.
  예: RL 코드 수정 → `spec-rl-agent.md` + 관련 figure 항목 / supervised 수정 →
  `spec-supervised.md` / 원본 TF 코드와 동치성 확인 → `../grid-cells/`의 해당 파일 /
  DMLab 레벨 → workspace CLAUDE.md의 lab 읽기 목록.
- **path-scoped rules**: `.claude/rules/*.md`는 `paths:`에 맞는 파일을 만질 때만 자동으로
  로드됩니다. 공식 문서도 "reducing noise and saving context space"를 이 용도로 권합니다.
  RL 파일을 건드리면 "spec-rl-agent.md를 먼저 읽고, 해당 Methods 문단 원문을
  `docs/paper/text/main.txt`에서 grep으로 확인하라"는 규칙이 자동으로 따라옵니다.
- **남기는 강제 규칙**은 하나뿐입니다: "논문을 따른다고 주장하는 코드를 쓰거나 바꿀 때는
  spec 노트의 출처를 따라 **원문 해당 문단/figure를 직접 열어** 확인한다." 전체 재독이
  아니라 해당 부분만 읽습니다.
- **figures.md**에는 텍스트로는 알 수 없는 사실을 figure별로 적어 둡니다(예: Fig. 2b cue는
  벽면 텍스처, Fig. 2f 곡선의 구간별 수치, Fig. 2h 점선/실선의 의미). "이 figure를 다시
  열어야 하는 작업"도 함께 명시합니다.
- **spec 노트는 검증하고 나서 씁니다.** 작성 후 별도 에이전트가 노트와 원문 페이지를
  한 줄씩 대조해 불일치를 표시하고, 그걸 고친 뒤에 커밋합니다. 참고 repo의
  `docs/Banino et al. - 2018 - supervised-learning-hyperparameters.md`가 좋은 예입니다
  (표마다 PDF 페이지 출처, pdftotext가 `10^-5`를 `10 5`로 깨뜨리는 문제 같은 주의 사항).
- **workspace `CLAUDE.md`**는 repo 밖에 있어서 버전 관리가 안 됩니다. 프로젝트 지식은
  `grid-cells-torch/CLAUDE.md`와 `docs/`로 옮기고, workspace 쪽은 "여러 repo 개요 +
  lab 읽기 목록 + 각 repo의 CLAUDE.md를 보라"만 남겨 짧게 줄입니다.
- `supplements/`의 저자 노트(`*.hwpx`)와 `images/`는 그대로 둡니다(수정 금지 대상).

### 3.3 출처 명시 (DeepMind `grid-cells/README.md` 방식)

DeepMind README는 첫머리에 논문 링크("as [published in Nature](https://www.nature.com/articles/s41586-018-0102-6)")와
인용 요구 문구("must cite 'Banino et al. ... Nature 557.7705 (2018): 429.'")를 둡니다.
같은 방식으로 다음을 추가합니다.
- **README.md 맨 위 `## Paper` 절**: 저자 전체(Banino, Barry, Uria, Blundell, Lillicrap,
  Mirowski, Pritzel, Chadwick, Degris, Modayil, Wayne, Soyer, Viola, Zhang, Goroshin,
  Rabinowitz, Pascanu, Beattie, Petersen, Sadik, Gaffney, King, Kavukcuoglu, Hassabis,
  Hadsell, Kumaran — 작성 시 PDF 첫 페이지와 대조), 제목, *Nature* 557, 429–433 (2018),
  DOI `10.1038/s41586-018-0102-6` 링크, BibTeX 블록. "이 repo는 원저자와 무관한 독립
  재구현이며, 논문과 원본 코드의 저작권은 각 저작권자에게 있다"는 고지. 원본 코드
  [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells)(Apache-2.0) 명시.
- **`docs/paper/README.md`**: 같은 인용 정보 + 이 폴더 노트들이 "논문을 요약한 2차 자료"라는 점.
- **PDF 자체의 커밋 여부**: 출처를 밝힌다고 재배포 권한이 생기지는 않습니다(Nature 논문
  PDF와 Supplementary는 출판사 저작물). 그래서 기본값은 **PDF, figure PNG, 텍스트 추출본은
  `.gitignore`, 인용 정보와 DOI 링크, 직접 쓴 노트만 커밋**입니다. 커밋을 원하시면
  `.gitignore`에서 한 줄만 지우면 됩니다. Phase 2를 시작할 때 한 번 더 확인받습니다.

---

## 4. Repo 구조 재편

### 4.1 가이드와 참고 repo에서 가져오는 것 / 가져오지 않는 것

**가져오는 것**
- 라이브러리를 `grid_cells/` 패키지로 옮기고, 루트에는 CLI 진입점만 둡니다.
- `run_scripts.sh`: 자주 쓰는 명령을 번호 붙여 나열한 치트시트입니다(참고 repo 방식).
  실행하면 목록을 출력합니다.
- `docs/maintenance.md`, `docs/plan/`, `docs/goals/`.
- pytest 기반 `tests/test_*.py`.
- 데이터셋 디렉터리마다 `meta.json`(생성 파라미터, seed, git commit) 기록.

**가져오지 않는 것 (근거)**
- 참고 repo의 하위 패키지(`cells/`, `data/`, `training/` ...): 가이드가 "20개 미만이면 평탄하게"
  라고 했고, 재편 후 모듈은 약 16개입니다.
- `config.yaml` + CLI override 엔진: 가이드가 dataclass 유지를 명시했습니다. 대신 진입점마다
  꼭 필요한 argparse 플래그만 둡니다.
- 가이드의 `slam.txt` 삭제 단계: 현재 repo에는 해당 파일이 없습니다(가이드가 다른 시점의
  repo를 보고 쓴 것으로 보입니다).
- 가이드의 노트북 설명("00: 환경 설정·데이터, 01: 모델·학습 ...")은 실제와 다릅니다. 실제로는
  00 데이터 생성, 01 CSV→npz, 02 학습, 03 경로적분 평가, 04 bottleneck gridness, 05 세포 유형,
  06 DMLab 빌드, 07 체크포인트 재생입니다. 아래 매핑은 실제 내용을 기준으로 합니다.

### 4.2 목표 구조

```
grid-cells-torch/
├── CLAUDE.md, README.md, LICENSE, requirements.txt, run_scripts.sh
├── config.py                       # 루트 유지 (가이드). A2C 설정 추가
├── generate_data.py                # 노트북 00+01 대체: 시뮬레이터 → npz shard 직행
├── train.py                        # 노트북 02 대체: supervised 학습 CLI
├── analyze_supervised.py           # 노트북 03/04/05 대체: --analysis path|gridness|cell-types|all
├── train_rl_agent.py               # --algo a2c|a3c (기본 a2c)
├── eval_rl_agent.py                # eval_episode_diversity.py + 노트북 07 통합
├── grid_cells/
│   ├── __init__.py
│   ├── data_generation.py          # 노트북 00의 Raudies-Hasselmo 시뮬레이터 + shard writer
│   ├── dataset.py                  # npz 로딩 (CSV 변환 코드 제거)
│   ├── model.py, ensembles.py, scores.py, figures.py, evaluate.py, vision.py, actor_critic.py
│   ├── training.py                 # 기존 train.py (라이브러리)
│   ├── rl_common.py                # world_to_meters, 관측 전처리, ego_vel 계산, 공용 env 생성
│   ├── rl_a3c.py                   # 기존 rl_train.py (동결된 legacy, 기준선 재현·비교용)
│   ├── rl_a2c.py                   # 신규: 동기 배치 학습기
│   ├── vec_env.py                  # 신규: DMLab 서브프로세스 벡터 환경
│   └── replay_buffer.py            # uint8 통합 버퍼 추가
├── tests/                          # pytest
├── docs/                           # §3.2
├── scripts/
│   ├── setup_dmlab.sh              # 노트북 06 대체 (§4.4)
│   └── dmlab_patches/
│       └── modern-toolchain.patch  # lab/ 작업 트리에 적용할 빌드 패치 (git diff 형식)
└── levels/, lab/, results/         # 변경 없음
```

**주의**: `git status`를 보면 `notebooks/01, 03, 04, 05`에 **커밋되지 않은 수정**이 있습니다.
삭제하기 전에 그 diff를 확인해 변환에 반영하고, 일단 커밋해서 기록에 남긴 다음 삭제합니다.

**노트북은 전부 삭제합니다.** 각 노트북의 로직은 진입점과 `grid_cells/`로, assert는
`tests/`로 옮긴 다음 삭제합니다. git 기록에는 남으므로 필요하면 언제든 꺼내 볼 수 있습니다.
변환표는 다음과 같습니다.

| 노트북 | 이동 위치 |
|---|---|
| 00 데이터 생성 | `grid_cells/data_generation.py` + `generate_data.py` |
| 01 CSV→npz | 삭제 (§5, 직행 경로로 불필요), assert는 `tests/test_data_generation.py` |
| 02 학습 | `train.py`, assert는 `tests/test_supervised_smoke.py` |
| 03/04/05 분석 | `analyze_supervised.py --analysis path\|gridness\|cell-types\|all` |
| 06 DMLab 빌드 | `scripts/setup_dmlab.sh` + 패치 파일 (§4.4) |
| 07 체크포인트 재생 | `eval_rl_agent.py --mode play` |

규칙(`docs/maintenance.md`): `grid_cells/` 안에는 `if __name__ == "__main__"`을 두지
않습니다(테스트로 강제). 새 설정은 `config.py` dataclass에 추가합니다. 결과물은
`results/`에 저장합니다.

### 4.3 진행 방식 — 동작 불변을 증명하면서 옮기기

1. **이동 전 golden 기록** (`tests/golden/*.npz`): 고정 seed로 (a) supervised `train_step`
   30회의 loss 열, (b) `GridCellsRNN`·`VisionModule`·`ActorCriticLSTM` forward 출력,
   (c) `scores.GridScorer`의 고정 ratemap 점수, (d) 기존 shard 0000의 앞 100개 궤적 해시.
2. `git mv`로 이동하고 import 경로만 수정합니다(`from model import` → `from grid_cells.model import`).
   로직은 건드리지 않습니다.
3. golden test가 비트 단위로(또는 float 허용오차 0으로) 일치하는지 확인합니다.
4. 진입점 작성: 노트북 셀의 assert(형상, 유한성, 초기 loss ≈ log256+log12 등)는 버리지 않고
   `tests/`로 옮깁니다.
5. `run_scripts.sh` 초안:
   ```
   1. python generate_data.py                       # 1,000,000 궤적 → data/datasets/<id>/
   2. python generate_data.py --num-shards 2 --tag trial
   3. python train.py                               # supervised 300k updates
   4. python analyze_supervised.py --checkpoint data/checkpoints/baseline/checkpoint_epoch299.pt --analysis all
   5. python train_rl_agent.py                      # A2C (GPU)
   6. python train_rl_agent.py --algo a3c           # legacy 기준선
   7. python train_rl_agent.py --level tiny_goal --total-env-steps 10000000   # 게이트 실행
   8. python eval_rl_agent.py --episodes 5
   9. pytest -m "not dmlab and not slow"            # 빠른 테스트
   10. pytest                                       # 전체
   11. bash scripts/setup_dmlab.sh --skip-deps      # DMLab 빌드/설치 (노트북 06 대체)
   ```
6. README는 새 명령에 맞춰 고치고 §3.3 출처 절을 맨 위에 추가하며, 논문 대조 내용은 `docs/paper/deviations.md`로 옮깁니다.
   README Known gaps #9의 오독(하이퍼파라미터 샘플링 단위는 actor가 아니라 replica이며,
   Methods에 "60 replicas were run with hyperparameters sampled from the same interval"이라고
   되어 있음)도 바로잡습니다.

### 4.4 노트북 06 → `scripts/setup_dmlab.sh`

**조사 결과.** 노트북 06은 이미 실제 상태보다 뒤처져 있습니다. 실제 `lab/` 작업 트리에는
노트북에 없는 변경이 더 있습니다. (1) `lab/BUILD`의 vm.pk3 genrule을 `mkdir -p` /
`ln -sf`로 바꾼 패치(재빌드 시 "already exists" 실패 대책). (2) 추적되지 않은
`lab/build_dmlab.sh`가 있는데, 여기에 노트북에 없는 실전 지식이 들어 있습니다: conda base의
`python3`가 시스템 python3를 가려서 Bazel이 numpy 없는 파이썬을 잡는 문제(PATH 재설정),
`./bazel-bin` 심볼릭 링크가 깨지는 문제(`bazel info bazel-bin`으로 회피), env1 경로 하드코딩.
이 스크립트는 submodule 안에 있어서 grid-cells-torch repo에 커밋되지 않은 상태입니다.

**설계**
- `scripts/dmlab_patches/modern-toolchain.patch`: 현재 `lab/`의 변경(`BUILD`, `WORKSPACE`,
  `bazel/glib.BUILD`, `bazel/libxml.BUILD`, `python/pip_package/__init__.py`, 새 파일
  `.bazelrc`, `.bazelversion`)을 git diff 하나로 묶습니다. 노트북의 `%%writefile` 6개와
  달리 리뷰하기 쉽고, `git apply`로 적용과 되돌리기가 모두 됩니다.
- `scripts/setup_dmlab.sh [--python PATH] [--skip-deps] [--skip-build]`, 단계별 출력은
  `[1/7]` 형식입니다.
  1. `git submodule update --init lab`
  2. 패치 적용(멱등). 이미 적용되어 있으면(`git apply --reverse --check` 성공) 건너뛰고,
     적용 가능하면 적용하며, 둘 다 아니면 중단합니다.
  3. 시스템 의존성: `apt-get install ...`(노트북 목록). sudo가 필요하므로 스크립트가 직접
     실행하지 않고 **명령을 출력한 뒤, 이미 설치돼 있는지(`dpkg -s`) 확인만** 합니다.
     사용자가 `! sudo apt-get install ...`로 직접 실행합니다.
  4. bazelisk 설치 여부를 확인합니다(없으면 `~/.local/bin`에 받음, sudo 불필요).
  5. PATH를 정리한 다음(`build_dmlab.sh`의 conda 회피 로직) `bazel shutdown` → `bazel build`.
  6. `bazel info bazel-bin`으로 wheel을 만들고 `--python`이 가리키는 환경에 pip install
     (기본값은 `conda run -n env1 which python`. 하드코딩 제거) + `dm_env`.
  7. 스모크 테스트(빈 방 20스텝, CUDA 텐서 변환), 그리고 `levels/` 로딩 확인.
- `lab/build_dmlab.sh`와 빌드 로그 2개는 내용이 새 스크립트로 흡수된 것을 확인한 뒤 삭제
  여부를 묻습니다.

**검증**
- 패치 파일이 현재 상태를 정확히 재현하는지: `lab`의 pinned commit으로 임시 worktree를
  만들고 패치를 적용한 결과가 현재 `lab/` 작업 트리와 diff 0인지 확인합니다.
- `--skip-deps --skip-build`로 1·2·4·7단계를 먼저 실행합니다(이미 설치된 env1에서 스모크 통과).
- 전체 재빌드 검증은 임시 worktree에서 한 번 수행합니다(Bazel 빌드는 수십 분 걸리므로
  tmux/백그라운드에서). 실패하면 기존 설치는 건드리지 않습니다.

---

## 5. 데이터 생성: 시뮬레이터 → npz 직행

**확인한 사실.** `00_make_dataset.ipynb`는 ratinabox에서 `Environment(params={'scale': 2.2})`의
`scale` 값 하나만 가져다 쓰고, 궤적은 노트북 안에 직접 구현한 Raudies–Hasselmo 시뮬레이터
(`simulate_rh_trajectory`)로 만듭니다. 즉 "ratinabox 데이터"는 사실상 자체 시뮬레이터
출력입니다. 따라서 ratinabox와 pandas 의존성을 뺄 수 있습니다(L=2.2를 config에서 읽음).

**설계** (`grid_cells/data_generation.py`, `generate_data.py`)
- `simulate_rh_trajectory`, `_wall_zone`, `_wall_redirect`는 노트북에서 그대로 옮깁니다.
- 궤적 하나에 대해 지금처럼 750 fine step → 100 step 보간(cubic/linear 동일)을 하고,
  `csv_shard_to_arrays`와 **같은 수식**으로 바로 `init_pos, init_hd, ego_vel=[speed, sin dθ, cos dθ],
  target_pos, target_hd`를 계산해 float32로 저장합니다.
- shard 단위 병렬(`mp.Pool`, shard마다 `np.random.seed(file_idx)`)도 기존과 같습니다.
  shard 하나는 1만 궤적, 기존 파일명 규칙(`0000-of-0099.npz`)도 유지합니다.
- 출력: `data/datasets/<id>/shards/*.npz` + `meta.json` + `data/latest` symlink(참고 repo 방식).
  `--num-shards`, `--tag`, `--output-dir`, `--workers` 플래그를 둡니다.
- `dataset.py`에서 `csv_shard_to_arrays`, `convert_all`, `COLUMNS`, pandas import를 지우고,
  `check_shards`, `GridCellsDataset`, `build_dataloader`는 유지합니다.

**검증 (가장 강한 테스트)**: 같은 seed로 새 경로가 만든 shard 0000이 **기존
`data/shards/0000-of-0099.npz`와 일치**해야 합니다. 기존 경로는 float64 → CSV 텍스트(pandas
repr, 왕복 손실 없음) → float32였으므로, 새 경로(float64 → float32)와 비트 단위로 같거나
최소 `atol=1e-6` 이내여야 합니다. 이 테스트가 통과하면 1,000,000 궤적을 다시 생성할 필요도
없습니다.

**기존 17GB CSV 디렉터리는** 검증 통과 후 삭제 여부를 사용자에게 물어봅니다(자동 삭제 안 함).

---

## 6. 논문 재대조에서 새로 찾은 의심 지점 (테스트 대상 추가)

Methods 원문(본문 PDF, "grid cell network of the agent" 문단, "actor–critic learner" 문단)과
`rl_train.py`를 대조한 결과입니다.

| ID | 코드 위치 | 현재 동작 | 논문 | 판정 |
|---|---|---|---|---|
| A | `rl_train.py:165` | bootstrap = V(s_t) | n-step 표준은 V(s_{t+n}) | 버그 |
| B | `rl_train.py:108` | reward 입력이 항상 0 | "concatenated with the reward r_t" | 버그 |
| C | `rl_train.py:155-156` | 텔레포트 후 grid LSTM을 **텔레포트 전 화면**의 vision 예측으로 재초기화 | "grid LSTM ... not initialized with ground truth ... instead set to zero", "had to use y and z to learn how to reset its internal state each time it was teleported" | 버그(재초기화 자체가 논문에 없고, 쓰는 입력도 틀림) |
| D | `rl_train.py:142-145` | 에피소드가 끝나 새 아레나가 되어도 `goal_grid_code`와 `prev_action`을 초기화하지 않음(hidden은 153~158행에서 리셋되지만 grid 쪽은 C의 잘못된 입력으로 리셋) | "goal grid code ... or zeros if the goal had not yet been reached in the episode" | 버그(goal code는 확정, prev_action은 표준 관행) |
| E | `rl_train.py:157, 165` | goal에 도착하면 policy LSTM을 리셋하고 bootstrap=0(종료로 취급) | goal 도착은 에피소드 종료가 아님(텔레포트 후 계속) | 논문이 명시하지 않음 → 표준 관행(에피소드 중간에는 리셋하지 않고 bootstrap 이어감)으로 변경, deviations에 기록 |
| F | `vision.py:47-50` | 마스크가 unit마다 독립(95%) | "provided as input ... 5% of the time on average" vs "zeroed the units with a probability of 95%" | 원문이 모호함. README는 per-unit으로 "고쳤다"고 기록함. 결정 근거를 deviations에 명시하고 config 플래그로 둠 |
| G | `replay_buffer.py` `SequenceAccumulator` | 텔레포트에서 시퀀스를 자름 | grid net이 "텔레포트 후 스스로 리셋하는 법"을 배우려면 텔레포트를 걸치는 시퀀스가 필요함 | C를 고치면 함께 바꿈(에피소드 경계에서만 자름) |

**n_step / action repeat 문제 (해결됨).** Methods의 "All threads applied updates to their
gradients every four actions (that is, action repeat of 4)"는 문장만 보면 "4행동마다 업데이트"로도
읽힙니다. 하지만 Supplementary Table 2(p.24)가 두 값을 따로 명시합니다:
"Back-propagation step in the actor-critic learner **100** — Number of backpropagation step used
to unroll the LSTM", "Action repeat **4** — Repeat each action selected by the agent this many times".
또 Supplementary 3c는 n-step return을 `R̂_t = Σ_{i=0..n-1} γ^i r_{t+i} + γ^n V(s_{t+n})`로
정의합니다. 그래서 현재 해석(행동 1회 = 4프레임, 100행동마다 BPTT 100 + 업데이트, 1e9는
프레임 수)을 유지합니다. 3c의 수식은 버그 A의 정답이 V(s_{t+n})임을 원문으로 확인해 줍니다.
spec-rl-agent.md에는 이 근거와 함께 Methods 문장이 모호하다는 사실을 기록합니다.

---

## 7. A2C 설계

### 7.1 구조

```
[메인 프로세스, GPU]                                  [env 프로세스 ×32, CPU]
 for t in 1..n_step(=100):
   images(B,3,64,64) ─ vision(no_grad) ─ grid.step(no_grad) ─┐
   actor_critic.step(batch B) → actions(B) ─────────────────▶ 각자 DMLab.step(a_i, 4프레임)
   ◀── obs, reward, episode_ended, pos/vel (B개) ──────────────
   per-env 리셋 마스크 적용 (§6 C/D/E 반영)
 bootstrap V(s_{t+n}) (no_grad, 다음 관측으로) ; 종료 env는 0
 AC 업데이트 1회 (B×100 transition 평균)
 vision 업데이트 K_v회, grid 업데이트 K_g회 (버퍼 샘플)
```

- `vec_env.py`: env마다 서브프로세스 하나와 `Pipe`. 명령은 `reset(seed)`, `step(action)`,
  `close`이고, 에피소드가 끝나면 자동으로 reset합니다. 시드는
  `base + env_idx + num_envs * episode_idx`로 결정적이고 env끼리 겹치지 않습니다(기존
  `seed + rank + step_counter`보다 재현성이 좋음). 관측은 **uint8 그대로** 전송합니다.
- `env_factory` 주입을 지원해 DMLab 없이 가짜 env로 전체 루프를 테스트할 수 있게 합니다.
- **설정 추가** (`config.py`): `algorithm="a2c"`, `num_envs=32`, `device="cuda"`,
  `vision_updates_per_rollout`, `grid_updates_per_rollout`, 그리고 §6 E/F/G의 동작 플래그
  (기본값은 위 판정을 따름).
- **vision/grid 업데이트 비율**: A3C에서는 별도 프로세스가 최대 속도로 돌았기 때문에 비율이
  정해져 있지 않았습니다. 지난 1e9 run 로그에서 측정한 비율은 vision 약 1회/250프레임,
  grid 약 1회/6,500프레임입니다(재시작으로 로그가 섞여 있어 근사치). 롤아웃 하나가
  32×100×4 = 12,800프레임이므로 기본값을 `K_v=51`, `K_g=2`로 두고, deviations에
  기록합니다.
- **Replay 버퍼**: 이미지를 uint8로 저장합니다(무손실, float32 대비 1/4).
  frame과 sequence를 따로 복사해 두지 않고, env별 링버퍼 하나에서 vision은 프레임을,
  grid는 같은 에피소드 안의 연속 100스텝 창을 뽑습니다. 이는 논문의 "shared buffer"
  구조와도 같습니다. CPU pinned memory에 두고, 샘플만 GPU로 보냅니다. 용량은 기존과 같은
  총 400k 프레임(약 4.9GB)에서 시작합니다.
- **체크포인트**: 기존 키 구조를 유지해 `eval_rl_agent.py`와 호환되게 합니다. 버퍼가
  uint8이라 full 체크포인트는 약 19.7GB에서 약 5GB로 줄어듭니다.
- `train_rl_agent.py`: `--algo a2c`이면 GPU를 쓰고, `CUDA_VISIBLE_DEVICES=""`는 a3c일 때만
  설정합니다(기존 CUDA 드라이버 크래시 대책은 A3C 전용).
- **A3C 코드(`rl_a3c.py`)는 동결해서 보존합니다.** 기준선을 재현하고, A2C가 게이트에서
  실패할 경우 "A3C + 버그 수정"과 비교하는 진단 도구로 씁니다.

### 7.2 하이퍼파라미터

처음에는 Table 2 값(lr, entropy, baseline cost의 구간 중앙값, discount 0.99, RMSProp)을 그대로
씁니다. A2C는 같은 프레임 수 동안 업데이트 횟수가 A3C의 1/32이므로, 게이트에서 학습 속도가
부족하면 **그때** lr을 구간 상한(2e-4) 쪽으로 옮기는 것을 첫 번째 조정 후보로 둡니다
(한 번에 하나씩).

### 7.3 처리량 게이트

예상치는 약 9,200 프레임/s(1e9 ≈ 30시간)입니다. **4,000 프레임/s 미만이면** 본 학습 전에
병목(env 쪽인지 GPU 쪽인지)을 프로파일링합니다. 기존 A3C는 약 1,000 프레임/s였습니다
(1e9에 11.6일).

---

## 8. 테스트 계획 (변경 반영)

**바뀐 점 요약**: (1) 독립 스크립트 + `run_all.py` 대신 **pytest**를 씁니다(참고 repo 방식,
`requirements.txt`에 추가). (2) import 경로가 `grid_cells.*`로 바뀝니다. (3) wiring test는
A3C(빨간불 확인용)와 A2C(초록불 목표) **양쪽에 같은 계약**을 적용합니다. (4) 데이터·재편·A2C
전용 테스트가 추가됩니다. (5) marker: `dmlab`(deepmind_lab이 없으면 skip), `slow`, `gpu`.

### 8.1 재편 안전망
- `test_golden.py` — §4.3의 golden과 일치.
- `test_package_rules.py` — 모든 모듈이 import되는지, `grid_cells/` 안에 `__main__` 블록이 없는지.

### 8.2 Known-answer test (기존 계획 유지 + 확장)
- `test_returns.py` — `rewards=[0,0,10], γ=0.99, bootstrap=2.0 → [11.741598, 11.8602, 11.98]`.
  **배치 + done 마스크 버전**도 추가합니다(예: env 0은 t=1에서 종료, env 1은 계속).
- `test_coords.py` — `world_to_meters`: 600→0, 1100→+1.25, 100→−1.25.
- `test_scores.py` — 합성 육각 격자는 0.37 초과, 노이즈는 약 0, 가우시안 한 덩어리와
  정사각 격자는 낮음, λ와 2λ의 scale 비율은 약 0.5.
- `test_ensembles.py` — posterior argmax, `decode_position` 세 모드.

### 8.3 데이터
- `test_data_generation.py` — (a) **새 경로로 만든 shard 0000 == 기존 shard 0000** (기존
  파일이 없으면 skip). (b) 궤적 몇 개에 대해 "옛 CSV 경로(오라클로 테스트 안에 보존)" ==
  "새 직행 경로". (c) `_wall_redirect`가 진입한 벽 쪽으로 다시 돌리지 않음(노트북의
  수치 검증을 테스트로). (d) 형상, dtype, 유한성(`check_shards`).
- `test_supervised_smoke.py` — 노트북 02의 assert들(형상, 초기 loss ≈ log256+log12,
  gradient 유한, 30스텝 진행).

### 8.4 Wiring test (가짜 env + 스파이 모듈)
같은 테스트 함수를 `actor_worker`(A3C)와 A2C collector 양쪽에 `parametrize`로 적용합니다.
A3C에서는 **실패해야** 하고(`xfail(strict=True)`로 "버그가 아직 있음"을 고정), A2C에서는
통과해야 합니다.

| 테스트 | 확인하는 것 | A3C 예상 |
|---|---|---|
| `test_reward_feedback` | t에 r=10이면 t+1의 `actor_critic.step(reward=10)` | 실패 (B) |
| `test_bootstrap_next_state` | 창 종료 시 bootstrap = f(obs_{n}) | 실패 (A) |
| `test_episode_reset` | 에피소드 종료 후 goal code·reward·prev_action·두 hidden이 0 | 실패 (D) |
| `test_teleport_semantics` | goal 텔레포트에서 grid hidden이 재초기화되지 않고, policy LSTM이 유지되며, bootstrap이 이어짐 | 실패 (C, E) |
| `test_sequence_spans_teleport` | grid 시퀀스가 텔레포트를 걸치고 에피소드 경계에서는 끊김 | 실패 (G) |

### 8.5 A2C 전용
- `test_batch_independence` — B개 env 배치 step = env별 batch-1 step(출력과 hidden 모두),
  일부 env만 리셋될 때 다른 env의 hidden이 그대로인지.
- `test_a2c_equals_a3c_math` — B=1에서 A2C loss/gradient == (버그 A를 고친) A3C 업데이트의
  loss/gradient.
- `test_vec_env_fake` — 가짜 env로 자동 reset과 시드 규칙 확인. `test_vec_env_dmlab`(dmlab
  marker)은 실제 DMLab 4개 env로 몇 스텝.
- `test_replay_uint8` — uint8 왕복이 무손실이고, 시퀀스 창이 에피소드 경계를 넘지 않는지.
- `test_toy_learning` (slow) — **DMLab 없이** 2~3개 행동의 가짜 env(정답 행동에 보상)에서
  A2C가 수백 업데이트 안에 정답 확률 0.9를 넘는지. 부호 오류나 advantage 붕괴 같은 문제를
  몇십 초 만에 잡는 첫 행동 게이트입니다.

### 8.6 진단 계측
`ac_diag.csv`: policy_loss, value_loss, entropy, mean_value, advantage 표준편차, "스텝당 이동
1cm 미만 비율", 처리량(프레임/s). 기존 `vision_loss.csv`/`grid_loss.csv` 규약을 따릅니다.

---

## 9. 실행 순서 (Phase)

| Phase | 내용 | 완료 조건 |
|---|---|---|
| 0 | plan을 repo `docs/plan/`에 복사, `STATUS.md` 작성 (계정 전환은 보류, §2.3) | 커밋 완료 |
| 1 | 기준선 동결: 1e9 A3C 결과 표, 영상, 체크포인트 보존 기록(`docs/plan/baseline-a3c.md`) | — |
| 2 | 컨텍스트 재편(§3): `docs/paper/` 이동, spec/figures/deviations 노트, 노트-원문 대조 검증, CLAUDE.md 재작성, rules | 검증 에이전트 불일치 0 |
| 3 | golden 기록 → 패키지 이동 → 진입점, `run_scripts.sh`, `maintenance.md`, pytest 도입, 노트북 삭제, `setup_dmlab.sh`(§4) | golden, 8.1, 패치 재현 diff 0 통과 |
| 4 | 데이터 직행 경로(§5) | 8.3 통과 (shard 0000 일치) |
| 5 | KAT와 wiring test 작성. A3C에서 A~G가 `xfail`로 빨간불인지 확인(§8.2, 8.4) | 빨간불 확인 |
| 6 | A2C 구현(§7): vec_env, collector, learner, uint8 버퍼, 체크포인트, CLI | 8.4(A2C)와 8.5 통과, 처리량 4,000 프레임/s 이상 |
| 7 | 게이트 ① `test_toy_learning` → ② `levels/tiny_goal.lua`(작은 방, goal이 보임) 1e7 프레임: 평균 reward가 초기 대비 5배 이상, value와 entropy가 붕괴하지 않음, 정지 비율이 포화되지 않음 | 통과 |
| 8 | 본 학습(square arena, **처음부터 새로**. A2C면 1e9 ≈ 30시간이라 A3C 가중치 재사용보다 깨끗함). 중간 게이트: 1e7 → 5 이상, 5e7 → 50 이상, 1e8 → 150 이상, 2.5e8 → 250 이상. 미달하면 즉시 중단. 각 게이트에서 `eval_rl_agent.py`로 영상과 궤적 확인 | 2.5e8 게이트 통과 |
| 9 | (병렬) gridness 7.8% 규명: `test_scores.py` 결과로 채점기 문제인지 네트워크 문제인지 판정 → annulus, `pc_scale`, 원형 아레나, bottleneck 폭 순으로 조사 | — |
| 10 | 논문 대조: 100 에피소드 평균 vs 289, Fig. 2h 직선 복귀, Fig. 2i fake goal, grid-like 비율 vs 21.4% | — |

**Phase 7 불합격 시 용의자 순서**: (1) lr 상한 쪽으로 이동(§7.2) → (2) §6 F 마스크 방식 →
(3) n_step을 더 짧게 (Table 2의 100에서 벗어나는 것이므로 진단 한정, 문서화) → (4) 디버깅 한정으로 grid/vision을 freeze해 정책만 학습
→ (5) 같은 조건에서 동결된 A3C에 A/B만 고친 버전을 돌려 A2C 구현 문제인지 판별 →
(6) entropy 계수 상향(문서화 후).

---

## 10. 수정 대상 파일 (대표)

- 신규: `grid_cells/{__init__,data_generation,rl_common,rl_a2c,vec_env}.py`, `generate_data.py`,
  `analyze_supervised.py`, `eval_rl_agent.py`, `run_scripts.sh`, `scripts/setup_dmlab.sh`,
  `scripts/dmlab_patches/modern-toolchain.patch`, `CLAUDE.md`(repo),
  `.claude/rules/*.md`, `docs/**`, `tests/test_*.py`, `tests/golden/`, `levels/tiny_goal.lua`
- 이동과 수정: 기존 루트 모듈 → `grid_cells/`(`train.py` → `training.py`,
  `rl_train.py` → `rl_a3c.py`), `train.py`(새 CLI), `train_rl_agent.py`(`--algo`),
  `config.py`(A2C 설정), `dataset.py`(CSV 코드 제거), `replay_buffer.py`(uint8 통합 버퍼),
  `vision.py`(마스크 플래그), `requirements.txt`(pytest 추가, ratinabox/pandas 제거),
  `.gitignore`, `README.md`
- workspace 루트 `CLAUDE.md`(축소), `supplements/논문`, `supplements/figures` → `grid-cells-torch/docs/paper/`

## 11. 리스크

- **A2C 전환과 버그 수정이 동시에 들어갑니다.** 원칙 3과 충돌하지만, A3C를 동결 보존해
  두었으므로 실패하면 "A3C + 수정"과 비교해 원인을 분리할 수 있습니다(§9 용의자 5).
- spec 노트로 압축하면 세부 사항이 빠질 수 있습니다 → 검증 에이전트 대조, figures.md,
  "주장하는 코드는 원문 확인" 규칙으로 막습니다.
- 랩실 조직이 server-managed settings로 모델이나 기능을 제한할 수 있습니다 → 전환 직후
  `/status`와 `/model`로 확인합니다.
- vision/grid 업데이트 비율은 논문에 없는 값입니다 → 측정값을 쓰고 문서화합니다.
