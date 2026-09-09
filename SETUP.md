# SETUP

Automated Cross-Consistency Assessment for General Morphological Analysis.

Installation and operation on an offline machine. Every step says what should
happen, how to confirm it did, and what to do when it does not.

**Read Part 0.** It takes two minutes and prevents most of Part 6.

---

## PART 0 — BEFORE YOU START

### 0.1 The short version

```bat
java -version                 must show 21 or higher
java Restore.java             unpack (no Python, no compile step)
start_offline.bat             run
```

Then open `http://localhost:8080`.

That is the whole procedure. If it worked, Part 6 is the only other section you
need, and only if something breaks.

### 0.2 What you are installing

**One process.** That is the entire system.

| | | |
|---|---|---|
| **Application** | Java, port 8080 | Serves the interface and the API, and runs both models inside itself |
| **Interface** | your browser | Served by the application, same address |

There is no second service, nothing to start in a particular order, and no
Python.

**Two models run, both inside the JVM, both on the CPU:**

| Model | Size | Drives | Works from |
|---|---|---|---|
| **llama.cpp + Qwen2.5-7B GGUF** | 4.68 GB | the interface's full analysis | value **names** |
| **Fine-tuned Qwen2.5-3B, ONNX 4-bit** | 4.27 GB | `/api/matrix/cca-score`, `/api/matrix/cca-model` | each value's **REQUIRES and PROVIDES** |

The second is the model this project produced: a rank-32 QLoRA adapter on
Qwen2.5-3B, merged into the base weights, exported to ONNX and quantised to
4-bit block-wise.

**Why two.** The fine-tuned model judges from what each value requires and
provides. Given bare names it scores at chance — that finding is what the whole
prompt format came from. The interface collects names only, so the interface is
served by llama.cpp and the fine-tuned model is reached by API. This is a
property of the model, not a defect.

### 0.2b If the machine has no Java

A JDK ships with the package as insurance:

```
jdk\OpenJDK21-jdk_x64_windows_hotspot.zip     205 MB, Temurin 21.0.5
```

Unzip it anywhere. Nothing needs installing and no administrator rights are
required — it is a directory, not an installer.

```bat
set JAVA_HOME=C:\gma\jdk\jdk-21.0.5+11
set PATH=%JAVA_HOME%\bin;%PATH%
java -version
```

Those two `set` lines apply to the current terminal only. To make them permanent,
add them under *Environment Variables* in Windows, or just run everything from a
terminal where you have set them.

Or skip PATH entirely and call it directly:

```bat
C:\gma\jdk\jdk-21.0.5+11\bin\java -Xmx8g -jar target\morphological-analyzer-0.0.1-SNAPSHOT.jar
```

If `java -version` already shows 21 or higher, ignore this and delete the folder.

### 0.3 Requirements

| | Minimum | Notes |
|---|---|---|
| **Java** | **21** | `java -version` must show 21+. **The only requirement**, and a JDK ships in `jdk\` if the machine has none — see 0.2b. |
| RAM | 16 GB | Both models are held resident. Start with `-Xmx8g`. |
| Disk | 20 GB free | Peak while unpacking; about 9 GB once the volumes are deleted. |
| OS | Windows 10/11 | Linux works; adjust path separators. |
| GPU | **none** | Not used and not supported. Everything runs on the CPU by design. |
| Python | **not needed** | Not to unpack, not to run. |
| Internet | **never** | Not at any point, before or after unpacking. |

### 0.4 What you should have received

| | |
|---|---|
| `gma_part_01.zip` … `gma_part_07.zip` | 7 volumes, 8.53 GB total |
| `MANIFEST.json` | sizes and SHA-256 for every file |
| `Restore.java` | the unpacker |
| `jdk\` | Temurin JDK 21, in case the machine has none (0.2b) |
| `nodejs\` | Node 18 and the Angular tooling, to rebuild the interface (8.4) |

All seven volumes are needed. Two large files are split across them and are
rejoined during unpacking.

---

## PART 1 — UNPACKING

### 1.1 Rejoin the archive parts

Put all nine files in **one folder** — the seven volumes, `MANIFEST.json` and
`Restore.java` together. The folder can be called anything and live anywhere;
every path used after unpacking is relative.

```
<any folder>  gma_part_01.zip ... gma_part_07.zip
  MANIFEST.json
  Restore.java
```

Then, from that folder:

```bat
java Restore.java
```

It unpacks **into that same folder**, creating `morphological-analyzer\`,
`training\`, `jdk\`, `nodejs\`, `SETUP.md` and `start_offline.bat` beside the
volumes. Delete the seven zips afterwards to reclaim 8.5 GB.

To unpack somewhere else instead:

```bat
java Restore.java --parts D:\gma --out C:\gma
```

No compile step and no Python. Java 11 and later run a single source file
directly, so this needs only the Java 21 runtime you already have.

It extracts every entry, rejoins the split files, and verifies each result
against the size and SHA-256 in the manifest.

**Expected ending:**

```
RESTORE COMPLETE -- 383 files, 0 mismatches
```

It is safe to run more than once.

**If it reports mismatches**, that file was corrupted in transfer. Re-copy the
affected volume and run it again. It reports every bad file rather than stopping
at the first, so one run tells you everything to re-copy.

**If it says "No gma_part_*.zip found"**, point `--parts` at the folder holding
the volumes.

### 1.2 Confirm the tree

```
morphological-analyzer\
├── SETUP.md                      this file
├── ARCHITECTURE.md               how it is built
├── PROJECT_REPORT.md             the report
├── Restore.java                  the unpacker
├── start_offline.bat             the launcher
│
├── morphological-analyzer\       the Java application
│   ├── target\morphological-analyzer-0.0.1-SNAPSHOT.jar    232 MB, runnable
│   ├── lib\                      123 dependency jars, for Eclipse (238 MB)
│   ├── .project .classpath       Eclipse project files (see Part 8)
│   ├── src\                      Java sources
│   ├── pom.xml                   Maven build
│   ├── data\overrides.json       analyst rulings (190 recorded)
│   └── models\
│       ├── cca-int4\             the fine-tuned model, ONNX 4-bit (4.27 GB)
│       ├── Qwen2.5-7B-Instruct-Q4_K_M.gguf                 4.68 GB
│       └── all-MiniLM-L6-v2\     embeddings (90 MB)
│
├── morphological-frontend\       Angular sources
└── training\                     corpus and training scripts (documentation)
```

**Two files must be exactly these sizes.** They are the split ones, so a partial
transfer shows up here first:

| File | Bytes |
|---|---|
| `models\Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4,683,074,240 |
| `models\cca-int4\model.onnx.data` | 4,267,737,088 |

`Restore.java` checks both, but if you skipped that step, check them by hand.

---

## PART 2 — STARTING THE SYSTEM

### 2.1 The launcher

```bat
start_offline.bat
```

On another port, optionally with a different thread count:

```bat
start_offline.bat 8090
start_offline.bat 8090 4
```

Open `http://localhost:8080`.

> **Do not use `start.bat` if you still have a copy.** It belongs to the old
> two-process layout: it starts a Python service that no longer exists, waits
> three minutes for it, then launches the application with
> `--gma.llm.enabled=false`, which switches off the interface's engine. The
> result starts cleanly and assesses nothing. It is not in this package.

### 2.2 Starting it by hand

```bat
cd morphological-analyzer
java -Xmx8g -jar target\morphological-analyzer-0.0.1-SNAPSHOT.jar ^
  --server.port=8080 ^
  --gma.llm.enabled=true ^
  --gma.llm.suppress-thinking=true ^
  --gma.llm.threads=8 ^
  --gma.onnxcca.threads=8
```

Three things matter here.

**Run it from the `morphological-analyzer` folder.** Both model paths are
relative to the working directory.

**`-Xmx8g`.** Both models are resident and the default heap is not enough on a
16 GB machine.

**Thread count.** Set it to your **physical** core count, not logical. Pinning it
also makes verdicts byte-reproducible: changing the thread count changes the
order of floating-point reductions and can flip a borderline cell.

### 2.3 Confirm what is actually running

```bat
curl -s http://127.0.0.1:8080/api/status
```

| Field | Expected | Meaning |
|---|---|---|
| `ccaEngine` | `llm` | the interface uses llama.cpp |
| `"llm":{"loaded":true}` | true | the 7B GGUF loaded |
| `"fineTunedCca":{"loaded":true}` | true | the fine-tuned model loaded |
| `symmetryAveraging` | true | both presentation orders are averaged |
| `overrideCount` | 190 | analyst rulings loaded |

Startup also prints:

```
+ ONNX CCA model loaded (CPU): cca-int4  verdict ids Y=809 N=451  threshold=0.5  threads=8
```

If `llm` is `loaded:false` see **6.9**. If `fineTunedCca` is `loaded:false` see
**6.10**. If `ccaEngine` is anything but `llm` see **6.11**.

### 2.4 First analysis

Enter a box in the interface and run the analysis.

**Expect about three and a half minutes** for a four-parameter, four-value box
on eight cores. The model is asked once per **parameter pair** — six times for
four parameters — not once per cell.

Measured on the aviation security screening box: 96 cells, 256 raw
configurations reduced to 72, in **203 seconds**.

Each excluded cell shows its provenance and evidence on hover. Cells the system
declines to decide are shown as referred, not silently defaulted either way.

---

## PART 3 — USING THE FINE-TUNED MODEL

The interface runs llama.cpp. The fine-tuned model needs characterisations, so
it is driven by API.

### 3.1 Score individual pairs

`POST /api/matrix/cca-score`

```json
{"pairs":[{
  "pa":"OPERATOR",   "a":"Private security contractor",
  "a_requires":"a commercial engagement",
  "a_provides":"personnel without sovereign powers",
  "pb":"LEGAL BASIS","b":"Territorial waters enforcement",
  "b_requires":"exercise of sovereign enforcement authority",
  "b_provides":"jurisdiction inside the territorial sea"
}]}
```

Reply:

```json
{"scores":[0.9866], "inconsistent":[true], "threshold":0.5,
 "engine":"onnx-int4-inprocess"}
```

`scores` is P(inconsistent), averaged over both presentation orders.

### 3.2 Assess a whole box

`POST /api/matrix/cca-model`

```json
{"parameters": {"OPERATOR": ["...","..."], "LEGAL BASIS": ["...","..."]},
 "characterisations": {"Private security contractor": {"requires":"...","provides":"..."}}}
```

Every value must have a characterisation. The endpoint refuses the request and
**names the ones missing** rather than scoring them from bare names, because a
score from a bare name is not a degraded result, it is a meaningless one.

Returns every cell with its score, verdict and detail, plus `cellsAssessed`,
`cellsExcluded`, `rawConfigurations` and `seconds`.

### 3.3 What it costs

**Roughly 4 seconds per cell** on eight cores. A four-parameter, four-value box
is 96 cells, so about **6–12 minutes** depending on the machine.

That is slower than the interface's engine because this model is asked once per
**cell**, twice over (both orders), rather than once per parameter pair.

### 3.4 Do not turn off symmetry averaging

Each cell is scored in both presentation orders and the results averaged.

This is not an optimisation that can be dropped for speed. Measured on the 1,191
held-out cells:

| | F1 at 70% coverage | F1 at 50% coverage |
|---|---|---|
| Both orders averaged | 0.693 | **0.811** |
| Single order | 0.241 | **0.000** |

Ranking survives (AUC 0.82) but the confidence signal **inverts**: the system
becomes least reliable exactly where it reports most certainty. There is no
setting to disable it, deliberately.

---

### 3.5 Running the baseline engine only

You may want to demonstrate the system without the fine-tuned model at all — if
it will not load, if the machine is short of memory, or simply to show the
baseline on its own.

```bat
start_offline.bat
```
then stop it, and start by hand with one extra flag:

```bat
cd morphological-analyzer
java -Xmx8g -jar target\morphological-analyzer-0.0.1-SNAPSHOT.jar ^
  --server.port=8080 ^
  --gma.llm.enabled=true ^
  --gma.llm.suppress-thinking=true ^
  --gma.llm.threads=8 ^
  --gma.onnxcca.enabled=false
```

**Understand what this actually changes**, because it is less than it sounds.

The interface has **never** used the fine-tuned model. It collects value names
only, and the fine-tuned model needs requires/provides, so the interface has
always been served by llama.cpp and the Qwen2.5-7B GGUF. Disabling the
fine-tuned model does not switch the interface to a different engine — it stops
loading a model the interface was not using.

| | Both models | `--gma.onnxcca.enabled=false` |
|---|---|---|
| Interface, full analysis | works | **works, identically** |
| Startup | 20–30 s | **17 s** |
| Resident memory | ~6.5 GB | **~4.8 GB** |
| Solution space on the test box | 256 → 72, 71.9% | **256 → 72, 71.9%** |
| `/api/matrix/cca-score` | works | **503 with the reason** |
| `/api/matrix/cca-model` | works | **503 with the reason** |

Measured, not estimated: the same box gives the same 71.9% reduction either way.

`/api/status` reports which mode you are in:

```json
"fineTunedCca":{"loaded":false,"error":"disabled by gma.onnxcca.enabled=false"}
```

The `error` field distinguishes a deliberate choice from a failure. If it says
`disabled by ...` you turned it off; anything else is a genuine load problem and
6.10 applies.

**The two engines are independent, not chained.** Neither falls back to the
other, and neither can take the other down. If the fine-tuned model fails to
load for any reason, the interface keeps working and a full assessment still
runs — this was verified by deliberately removing the model and running a
complete box, which produced an identical result.

To make the setting permanent rather than passing it each time, add
`--gma.onnxcca.enabled=false` to the `java` line inside `start_offline.bat`.

---

## PART 4 — CONFIGURATION

Settings live in `src/main/resources/application.properties` inside the jar, and
any of them can be overridden on the command line with `--name=value`.

### 4.1 The ones you might change

| Property | Default | Purpose |
|---|---|---|
| `server.port` | `8080` | Application port |
| `gma.llm.threads` | `8` | llama.cpp threads; use physical core count |
| `gma.onnxcca.threads` | `8` | ONNX threads; same |
| `gma.onnxcca.threshold` | `0.5` | P(inconsistent) above which a cell is excluded |
| `gma.llm.suppress-thinking` | `true` | Leave it true; see 6.8 |
| `gma.onnxcca.enabled` | `true` | Set `false` to run the baseline engine only; see 3.5 |
| `gma.overrides.file` | `data/overrides.json` | Analyst rulings |

### 4.2 The ones you should not

| Property | Default | Why |
|---|---|---|
| `gma.cca.engine` | `llm` | `finetuned` refers to a Python service that no longer exists. The application now **rejects** it with an explicit error rather than silently falling back |
| `gma.nli.enabled` | `false` | The NLI model measured **AUC 0.53** against expert labels — no better than a coin toss. It is not shipped |
| `gma.cca.url` | — | Pointed at the old Python service. Unused |
| `gma.onnxcca.model-dir` | `models/cca-int4` | Relative to the working directory |

### 4.3 Ports

Only one port is used now. Change it with `--server.port=8090`, or
`start_offline.bat 8090`.

The interface finds the API automatically on whatever port the server started
on — it resolves the API address from the page's own origin, so nothing needs
changing when the port changes.

To find what is holding a port:

```bat
netstat -ano | findstr :8080
tasklist /fi "pid eq <PID>"
```

---

## PART 5 — VERIFYING THE INSTALLATION

Five checks, in order. Each depends on the one before.

**5.1 Java is right**

```bat
java -version
```
→ `21` or higher.

**5.2 The files are intact**

```bat
java Restore.java
```
→ `RESTORE COMPLETE -- 383 files, 0 mismatches`

**5.3 The application starts and both models load**

```bat
curl -s http://127.0.0.1:8080/api/status
```
→ `"llm":{...,"loaded":true}` and `"fineTunedCca":{"loaded":true,...}`

**5.4 The fine-tuned model scores a pair**

POST the example in 3.1. A contractor against territorial-waters enforcement
scores **0.9866** on this build — the contractor has no sovereign authority and
that authority is exactly what the legal basis requires. Anything above the 0.5
threshold is the right answer; the exact figure can move slightly with thread
count, because that changes the order of floating-point reductions.

If everything scores near 0.5, the model loaded but something is wrong with the
input — check that `a_requires` and `a_provides` are populated.

**5.5 End to end**

Run a box in the interface. Expect a populated grid, a reduced solution space,
and evidence on hover for every excluded cell.

---

## PART 6 — WHEN THINGS GO WRONG

### 6.1 Anything asking for Python

Nothing in the running system uses Python, so this should not happen.

If you see it, you are running something from the old two-process layout —
`start.bat`, or `training\serve_cca.py` directly. Those scripts ship as the
record of how the model was built and evaluated; they are not part of running
it and nothing invokes them.

Unpacking is `java Restore.java`. Running is `start_offline.bat`.

### 6.2 `java` is not recognised

Java is not on PATH. Install a JDK 21, then **close every open terminal** — PATH
changes only reach terminals opened afterwards — and check again.

To use a JDK without touching PATH:

```bat
"C:\Program Files\Java\jdk-21\bin\java" -Xmx8g -jar target\morphological-analyzer-0.0.1-SNAPSHOT.jar
```

### 6.3 `UnsupportedClassVersionError`

The jar was built for Java 21 and you are running an older runtime. `java
-version` will show 17 or below. Install a JDK 21.

### 6.4 "It is running on the CPU"

That is correct and intended. There is no GPU path to fall back from.

llama.cpp is loaded with zero GPU layers and the ONNX components are created
with `CPUExecutionProvider` explicitly. Messages like

```
warning: one possible reason is that llama.cpp was compiled without GPU support
CUDA is not supported OnnxRuntime engine: Failed to find CUDA shared provider
```

appear on every start and are **not errors**. Nothing is degraded by them; a
CUDA device, where one exists, is simply unused.

If it is slow, see 6.8. The cause is never a missing GPU.

### 6.5 Out of memory

This is system RAM, not video memory.

| Symptom | Cause | Fix |
|---|---|---|
| `OutOfMemoryError: Java heap space` | Heap too small | `-Xmx8g`, as the launcher sets |
| Heavy swapping during a run | Under 16 GB RAM, or other applications resident | Close other applications; the models need about 6 GB between them |
| Process killed with no message | Windows commit limit | Check the pagefile is system-managed and at least 16 GB |

### 6.6 Port already in use

```
Web server failed to start. Port 8080 was already in use.
```

```bat
start_offline.bat 8090
```

The interface follows automatically.

### 6.7 The page loads but nothing happens

Open the browser console (F12).

| Console shows | Cause | Fix |
|---|---|---|
| CORS error | The interface is served from somewhere other than the application | Serve it from the application, or set `window.GMA_API_BASE` in `assets/config.js` |
| 404 on `/api/...` | Wrong port | Check the address bar matches the port the server reported |
| Nothing, and the grid stays empty | The analysis is still running | A box takes minutes; watch the server window |

### 6.8 It is very slow

Work through these in order. The first two are worth far more than the rest.

**1. Thinking is not suppressed.** The single largest cost. The model
deliberates by default, which on CPU is hundreds of extra tokens for a
sixteen-cell answer — measured at **515 s for one matrix against 45 s** with it
off.

```
--gma.llm.suppress-thinking=true
```

`start_offline.bat` passes this already. If you launch by hand, pass it.

**2. The thread count is unpinned or wrong.** Pin it to your **physical** core
count:

```
--gma.llm.threads=8 --gma.onnxcca.threads=8
```

**3. Everything else.**

| Cause | Check | Fix |
|---|---|---|
| Waiting on a Python service | Startup pauses ~3 min before anything happens | You are running `start.bat`. Use `start_offline.bat` |
| First run of a session | 4.7 GB read from disk | Expected once; later runs skip it |
| On battery | Windows throttles the CPU | Plug in |
| Another heavy process | Task Manager | The models want all pinned cores |

**Measured reference.** Eight cores, thinking suppressed: a four-parameter,
four-value box — 96 cells, six generations — completes in **203 seconds**. If
you are seeing five minutes or more, item 1 or item 3's first row is almost
certainly why.

The box is assessed once per *parameter pair*, not once per cell, so adding
values to an existing parameter costs far less than adding a parameter.

### 6.9 `"llm":{"enabled":true,"loaded":false}`

The interface's engine is on but the model file did not load.

| Cause | Fix |
|---|---|
| The GGUF is missing or truncated | It must be **4,683,074,240** bytes. Split across volumes; re-run `java Restore.java` |
| Not enough free RAM | It needs about 5 GB resident. Close other applications |
| Wrong working directory | `gma.llm.model` is relative. Launch from `morphological-analyzer`, as the launcher does |

If it says `"enabled":false`, something passed `--gma.llm.enabled=false` — that
is what `start.bat` does. Use `start_offline.bat`.

### 6.10 `"fineTunedCca":{"loaded":false}`

The interface still works; llama.cpp is a separate engine. What stops working is
`/api/matrix/cca-score` and `/api/matrix/cca-model`, which return **503** with
the reason.

| Cause | Fix |
|---|---|
| The model did not unpack | `models\cca-int4` needs **both** `model.onnx` and `model.onnx.data` (4,267,737,088 bytes). Re-run `java Restore.java` |
| `tokenizer.json` missing | It ships beside the model. Without it the service refuses to load rather than tokenise wrongly |
| Not enough heap | Start with `-Xmx8g` |
| Wrong `gma.onnxcca.model-dir` | Relative to the working directory |

**First check whether you turned it off.** If `/api/status` shows

```json
"fineTunedCca":{"loaded":false,"error":"disabled by gma.onnxcca.enabled=false"}
```

then nothing is wrong — that is the baseline-only mode described in 3.5.

Otherwise the startup log names the reason:

```
x ONNX CCA model failed to load: <reason>
```

**One failure is deliberate.** If the verdict tokens do not resolve to exactly
one token after `VERDICT:`, the service **refuses to load**. A service that
scores confidently from the wrong logit position is worse than one that does not
start.

### 6.11 `gma.cca.engine=finetuned` error on analysis

```
gma.cca.engine=finetuned refers to the Python scoring service, which no longer
exists. Use gma.cca.engine=llm.
```

Exactly what it says. That setting used to send the request to `serve_cca.py`
over HTTP. The service is gone, and rather than let the setting produce a silent
fallback that looks like a result, it is rejected.

Remove the override, or pass `--gma.cca.engine=llm`.

### 6.12 Results look wrong

**Everything consistent (nothing excluded).** Check `/api/status` shows an
engine loaded. If `ccaEngine` is `llm` and `"llm":{"loaded":true}`, the model is
running and this is its judgement — the criterion is deliberately permissive, a
pair is excluded only where no configuration could contain both.

**Everything inconsistent.** Almost always a threshold problem. Check
`gma.onnxcca.threshold` is `0.5`, not something near zero.

**The solution space is empty.** One parameter's values were all excluded against
another's, which eliminates every configuration. Look along the rows in the grid
for a parameter with no surviving pair; that is usually a value worded too
narrowly.

### 6.13 Machine sleeps during a long analysis

A box takes minutes and an unattended machine may sleep partway through. The
request then fails or the connection drops.

Keep it plugged in and set Windows to never sleep on AC. Software cannot
override a critical-battery sleep.

---

## PART 7 — DAILY OPERATION

### 7.1 Start and stop

```bat
start_offline.bat
```

Closing the console window stops it. Or:

```bat
netstat -ano | findstr :8080
taskkill /F /PID <pid>
```

### 7.2 What persists

| | Where | Survives restart |
|---|---|---|
| Analyst rulings | `data\overrides.json` | yes |
| Calibration head | `data\calibration.json` | yes, if trained |
| The box you were working on | browser local storage | yes, same browser |
| Assessment results | not stored | no — re-run the analysis |

Analyst rulings outrank both models on the pairs they cover, and survive re-runs.
Back up `data\overrides.json` if the rulings matter.

---

## PART 8 — OPENING IT IN ECLIPSE

**Import it as a plain Java project, not a Maven project.**

```
File → Import → General → Existing Projects into Workspace
Select root directory: ...\morphological-analyzer
Finish
```

**Do not use "Existing Maven Projects".** That makes Eclipse resolve 123
artifacts from the internet, which fails on an offline machine and leaves the
project full of unresolved-import errors.

The package ships `.project`, `.classpath`, `.settings` and a `lib\` folder
holding all 123 dependency jars. The classpath references them directly, so the
project resolves and compiles with **no Maven, no m2e and no network**. This was
verified by compiling the restored tree with `javac -cp "lib/*"`.

`pom.xml` is still present and is still authoritative for a command-line build,
if you ever have internet.

### 8.1 The classes worth walking through

| Class | What it does |
|---|---|
| `AiController` | All `/api/*` endpoints; builds the grid and the solution space |
| `OnnxCcaService` | The fine-tuned model. Prompt construction, one forward pass, logit read, both-orders averaging |
| `LlmCcaService` | llama.cpp through a JNI binding; drives the interface |
| `ConsistencyService` | Verdict precedence: analyst ruling, then model, then fallbacks |
| `OverrideStore` | Standing analyst rulings, and why they outrank the models |
| `CalibrationService` | The optional calibration head (inactive by default) |
| `OnnxEmbeddingService` | MiniLM embeddings, used only by the calibration head |
| `PairKey` | Order-independent key for a cell — cross-consistency is symmetric |

### 8.2 Running it from Eclipse

Run `MorphologicalAnalyzerApplication` as a Java Application, with:

- **VM arguments:** `-Xmx8g`
- **Working directory:** the `morphological-analyzer` folder — both model paths
  are relative to it, and this is the most common reason a run from an IDE loads
  no models when the launcher works fine.

### 8.3 Rebuilding the application jar

**This is the one thing you cannot do offline.**

```bat
mvnw clean package -DskipTests
```

`mvnw` does not carry Maven, it *downloads* it, and then Maven downloads 123
dependencies. On a machine with no internet it fails with
`NoPluginFoundForPrefixException`.

You almost certainly do not need it. Eclipse compiles and runs the application
without Maven (8.2). Maven is required only to produce a new **distributable
jar** — a file to hand to someone else. For editing, running and demonstrating,
Eclipse alone is enough.

### 8.4 Rebuilding the interface — this DOES work offline

Everything needed ships in `nodejs\`.

**One-time setup:**

```bat
cd nodejs
tar -xf node-v18.20.8-win-x64.zip
cd ..\morphological-frontend
tar -xf ..
odejsngular-node_modules.zip
```

`tar` is built into Windows 10 and 11; any unzip tool does the same job.

That gives you `node_modules\` with Angular 15, the Angular CLI and the build
chain — 566 packages, already resolved. **No `npm install` and no network.**

**Build:**

```bat
set PATH=C:\gma
odejs
ode-v18.20.8-win-x64;%PATH%
cd morphological-frontend
npm run build
```

Output lands in `dist\morphological-frontend\`.

**Then put it where the application serves it from:**

```bat
copy /Y dist\morphological-frontend\*.* ..\morphological-analyzer\src\main
esources\staticxcopy /E /I /Y dist\morphological-frontendssets ..\morphological-analyzer\src\main
esources\staticssets
```

This step is easy to forget. The application serves the interface from
`src/main/resources/static`, not from `dist/`, so a build alone changes nothing
you can see. After copying, restart from Eclipse and the new interface is live —
no jar rebuild needed, because Eclipse serves the resources directly.

**Verified:** the interface was rebuilt from these two archives alone, with the
bundled Node and no network, producing byte-identical bundles
(`main.1b5f976a733ee376.js`).

**A caution.** `npm install`, `npm update` or `ng update` will all try to reach
the network and will fail. The shipped `node_modules` is complete; leave it
alone.

## PART 9 — QUICK REFERENCE

```
Unpack                   java Restore.java
Start                    start_offline.bat
Start on another port    start_offline.bat 8090
Start by hand            java -Xmx8g -jar target\morphological-analyzer-0.0.1-SNAPSHOT.jar
Check both models        curl http://localhost:8080/api/status
Score a pair             POST /api/matrix/cca-score
Assess a box             POST /api/matrix/cca-model
Baseline engine only     --gma.onnxcca.enabled=false
Free a port              netstat -ano | findstr :8080
Interface                http://localhost:8080
```

**The four things that cause most problems**

1. **Using `start.bat`** — it starts a Python service that no longer exists,
   waits three minutes, then disables the interface's engine. Use
   `start_offline.bat`.
2. **Launching from the wrong folder** — both model paths are relative to
   `morphological-analyzer`. From anywhere else, neither model loads.
3. **Importing into Eclipse as a Maven project** — 123 artifacts it cannot
   download. Import as an existing project.
4. **Too little heap** — `-Xmx8g`. The default is not enough for two resident
   models.

**Expected timings, eight cores**

| | |
|---|---|
| Startup, both models | 20–30 s |
| Full box, interface engine | ~203 s |
| One cell, fine-tuned model | ~4 s |
| Full box, fine-tuned model | 6–12 min |
