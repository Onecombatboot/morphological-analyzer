package com.drdo.issa.morphological_analyzer;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class AiController {

    @Autowired
    private OnnxNliService onnxNliService;

    @Autowired
    private OnnxEmbeddingService onnxEmbeddingService;

    @Autowired
    private ConsistencyService consistencyService;

    @Autowired
    private OverrideStore overrideStore;

    @Autowired
    private CalibrationService calibrationService;

    @Autowired
    private LlmCcaService llmCcaService;

    @Autowired
    private OnnxCcaService onnxCcaService;

    /** 'finetuned' (the model trained for this task) or 'llm' (the GGUF baseline). */
    @org.springframework.beans.factory.annotation.Value("${gma.cca.engine:finetuned}")
    private String ccaEngine;

    // Helper class for matrix value mapping
    public static class ValueItem {
        public String parameter;
        public String value;

        public ValueItem(String parameter, String value) {
            this.parameter = parameter;
            this.value = value;
        }
    }

    public static class FullAnalysisResponse {
        public List<String> parameters;
        public Map<String, List<String>> parameterValues;
        public List<ValueItem> allValues;
        public String[][] ccaMatrix; // "X", "", or "BLACK"
        /** Provenance per cell: "OVERRIDE", "CALIBRATED", "MODEL" or "" for blacked-out cells. */
        public String[][] ccaSource;
        /** The number each decision was made on, for the hover explanation. */
        public double[][] ccaScore;
        /** A sentence explaining each decision, shown on hover. */
        public String[][] ccaDetail;
        public List<Map<String, String>> solutionSpace;
        /**
         * Cartesian product size. A long rather than an int: sixteen parameters of five values
         * overflows an int, and a silently negative problem space is worse than a large one.
         */
        public long totalProblemSpace;
        public int solutionSpaceSize;
        public double reductionPercentage;
        /** How many cells came from each source, so the analyst can see how much is their own. */
        public Map<String, Integer> sourceCounts;
    }

    @PostMapping("/matrix/full-analysis")
    public ResponseEntity<FullAnalysisResponse> runFullAnalysis(@RequestBody Map<String, List<String>> paramMap) {
        FullAnalysisResponse response = new FullAnalysisResponse();
        response.parameterValues = paramMap;
        response.parameters = new ArrayList<>(paramMap.keySet());

        // 1. Flatten all values into an ordered list for the CCA grid
        List<ValueItem> allValues = new ArrayList<>();
        long totalSpace = 1;
        for (String param : response.parameters) {
            List<String> vals = paramMap.get(param);
            totalSpace *= Math.max(1, vals.size());
            for (String val : vals) {
                allValues.add(new ValueItem(param, val));
            }
        }
        response.allValues = allValues;
        response.totalProblemSpace = totalSpace;

        int size = allValues.size();
        String[][] matrix = new String[size][size];
        String[][] source = new String[size][size];
        double[][] score = new double[size][size];
        String[][] detail = new String[size][size];
        Set<String> contradictionPairs = new HashSet<>();
        Map<String, Integer> sourceCounts = new LinkedHashMap<>();
        sourceCounts.put(ConsistencyService.SOURCE_OVERRIDE, 0);
        sourceCounts.put(ConsistencyService.SOURCE_LLM, 0);
        sourceCounts.put(ConsistencyService.SOURCE_CALIBRATED, 0);
        sourceCounts.put(ConsistencyService.SOURCE_MODEL, 0);

        // The model is asked once per parameter pair, with every value of both parameters visible,
        // rather than once per cell. Judging cells in isolation was what let the same concept get
        // opposite verdicts on trivial wording differences.
        // This flow is served by llama.cpp, in process. It works from value names,
        // which is all this endpoint carries.
        //
        // The fine-tuned model is NOT reachable from here and that is deliberate: it
        // judges from each value's REQUIRES and PROVIDES, and given bare names it
        // scores at chance. It is served by /api/matrix/cca-score and
        // /api/matrix/cca-model, which do carry characterisations.
        //
        // gma.cca.engine=finetuned used to send this to a Python service over HTTP.
        // That service is gone. Rather than let the setting silently produce a
        // fallback that looks like a result, it is rejected outright.
        if ("finetuned".equalsIgnoreCase(ccaEngine)) {
            throw new IllegalStateException(
                    "gma.cca.engine=finetuned refers to the Python scoring service, which no "
                    + "longer exists. Use gma.cca.engine=llm. The fine-tuned model runs in "
                    + "process and is reached through /api/matrix/cca-score and "
                    + "/api/matrix/cca-model, which require each value's requires/provides.");
        }
        Map<String, LlmCcaService.PairVerdict> llmVerdicts =
                llmCcaService.assessMatrix(response.parameters, paramMap);

        // Embeddings are only needed when a calibration head is active, but when it is, the same
        // handful of values recur across the whole grid, so each is embedded once.
        Map<String, float[]> embeddingCache = new HashMap<>();

        // 2. Build Table 4.2 Cross-Consistency Assessment (CCA) Grid
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                ValueItem rowItem = allValues.get(row);
                ValueItem colItem = allValues.get(col);

                // Black out upper triangle, diagonal, and intra-parameter comparisons
                if (col >= row || rowItem.parameter.equals(colItem.parameter)) {
                    matrix[row][col] = "BLACK";
                    source[row][col] = "";
                    detail[row][col] = "";
                } else {
                    // The NLI framing is directional, so always present the pair in the order the
                    // parameters were declared. The grid itself only walks its lower triangle and
                    // would otherwise hand over roughly half of all pairs back to front.
                    ValueItem first = rowItem;
                    ValueItem second = colItem;
                    if (response.parameters.indexOf(colItem.parameter)
                            < response.parameters.indexOf(rowItem.parameter)) {
                        first = colItem;
                        second = rowItem;
                    }

                    ConsistencyService.Verdict verdict = consistencyService.assess(
                            first.parameter, first.value,
                            second.parameter, second.value,
                            embeddingCache, llmVerdicts);

                    source[row][col] = verdict.source;
                    score[row][col] = verdict.score;
                    detail[row][col] = verdict.detail;
                    sourceCounts.merge(verdict.source, 1, Integer::sum);

                    if (verdict.contradiction) {
                        matrix[row][col] = "X";
                        contradictionPairs.add(PairKey.of(
                                rowItem.parameter, rowItem.value,
                                colItem.parameter, colItem.value));
                    } else {
                        matrix[row][col] = "";
                    }
                }
            }
        }
        response.ccaMatrix = matrix;
        response.ccaSource = source;
        response.ccaScore = score;
        response.ccaDetail = detail;
        response.sourceCounts = sourceCounts;

        // 3. Extract Table 4.3 Solution Space (Non-contradictory configurations)
        List<Map<String, String>> solutionSpace = new ArrayList<>();
        generateValidSolutions(response.parameters, 0, new LinkedHashMap<>(), paramMap, contradictionPairs, solutionSpace);

        response.solutionSpace = solutionSpace;
        response.solutionSpaceSize = solutionSpace.size();
        response.reductionPercentage = totalSpace > 0 ? (1.0 - ((double) solutionSpace.size() / totalSpace)) * 100.0 : 0.0;

        return ResponseEntity.ok(response);
    }

    /** Diagnostic: run an arbitrary premise/hypothesis, for comparing candidate framings. */
    @PostMapping("/debug/pair")
    public ResponseEntity<Map<String, Object>> debugPair(@RequestBody Map<String, String> req) {
        float[] logits = onnxNliService.rawPair(req.get("premise"), req.get("hypothesis"));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("logits", logits);
        return ResponseEntity.ok(out);
    }

    /**
     * Cross-consistency by the fine-tuned model, which needs more than value names.
     *
     * <p>Body: {@code {"domain":..., "parameters":{P:[v,...]}, "characterisations":{v:{"requires":
     * ..., "provides":...}}, "threshold":0.485}}. Every value named in {@code parameters} must
     * appear in {@code characterisations} — the model scores from what a value requires and
     * provides, not from its name, and the scoring service rejects the request otherwise with the
     * offending values listed.
     *
     * <p>Returns the verdict grid for every parameter pair, the solution-space reduction, and per
     * cell either a clause citation (deterministic rule) or P(inconsistent) (model), so an
     * exclusion can always be explained.
     */
    @PostMapping("/matrix/cca-model")
    public ResponseEntity<Map<String, Object>> ccaModel(@RequestBody Map<String, Object> body) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (!onnxCcaService.isLoaded()) {
            out.put("error", "the fine-tuned model is not loaded: "
                    + (onnxCcaService.getLoadError() == null
                       ? "no reason recorded" : onnxCcaService.getLoadError()));
            out.put("hint", "the interface's own analysis does not use this model and is "
                    + "unaffected; see SETUP.md 6.10");
            return ResponseEntity.status(503).body(out);
        }

        Object pv = body.get("parameters");
        Object ch = body.get("characterisations");
        if (!(pv instanceof Map) || !(ch instanceof Map)) {
            out.put("error", "expected \"parameters\" {name:[values]} and "
                    + "\"characterisations\" {value:{requires,provides}}");
            return ResponseEntity.badRequest().body(out);
        }
        Map<?, ?> params = (Map<?, ?>) pv;
        Map<?, ?> chars = (Map<?, ?>) ch;

        List<String> names = new ArrayList<>();
        for (Object k : params.keySet()) names.add(String.valueOf(k));

        // Every value must carry its characterisation. Scoring one without is
        // not a degraded result, it is a meaningless one: the model judges from
        // REQUIRES and PROVIDES, and given bare names it scores at chance.
        List<String> missing = new ArrayList<>();
        for (String p : names) {
            for (Object v : (List<?>) params.get(p)) {
                if (!chars.containsKey(String.valueOf(v))) missing.add(String.valueOf(v));
            }
        }
        if (!missing.isEmpty()) {
            out.put("error", "no characterisation for: " + String.join(", ", missing));
            return ResponseEntity.badRequest().body(out);
        }

        long t0 = System.currentTimeMillis();
        List<Map<String, Object>> cells = new ArrayList<>();
        int excluded = 0;
        for (int i = 0; i < names.size(); i++) {
            for (int j = i + 1; j < names.size(); j++) {
                String pa = names.get(i), pb = names.get(j);
                for (Object av : (List<?>) params.get(pa)) {
                    for (Object bv : (List<?>) params.get(pb)) {
                        String a = String.valueOf(av), b = String.valueOf(bv);
                        Map<?, ?> ca = (Map<?, ?>) chars.get(a);
                        Map<?, ?> cb = (Map<?, ?>) chars.get(b);
                        double p = onnxCcaService.scorePair(
                                pa, a, str(ca, "requires"), str(ca, "provides"),
                                pb, b, str(cb, "requires"), str(cb, "provides"));
                        boolean bad = onnxCcaService.isInconsistent(p);
                        if (bad) excluded++;
                        Map<String, Object> c = new LinkedHashMap<>();
                        c.put("parameterA", pa); c.put("valueA", a);
                        c.put("parameterB", pb); c.put("valueB", b);
                        c.put("score", p);
                        c.put("inconsistent", bad);
                        c.put("detail", String.format(
                                "Fine-tuned model scored %.3f P(inconsistent) against a "
                                + "threshold of %.2f, averaged over both presentation orders",
                                p, onnxCcaService.getThreshold()));
                        cells.add(c);
                    }
                }
            }
        }

        long raw = 1;
        for (String p : names) raw *= Math.max(1, ((List<?>) params.get(p)).size());

        out.put("cells", cells);
        out.put("cellsAssessed", cells.size());
        out.put("cellsExcluded", excluded);
        out.put("rawConfigurations", raw);
        out.put("threshold", onnxCcaService.getThreshold());
        out.put("engine", "onnx-int4-inprocess");
        out.put("symmetryAveraging", true);
        out.put("seconds", (System.currentTimeMillis() - t0) / 1000.0);
        return ResponseEntity.ok(out);
    }

    /**
     * Score individual value pairs with the fine-tuned model, in process.
     *
     * <p>Request shape is unchanged from the Python service this replaces:
     * <pre>
     * {"pairs":[{"pa","a","a_requires","a_provides",
     *            "pb","b","b_requires","b_provides"}, ...]}
     * </pre>
     * and the reply is still {@code {"scores":[...]}}, so existing callers do
     * not change. What changed is that nothing leaves the JVM.
     *
     * <p>The characterisations are required, not optional. This model judges
     * from what each value REQUIRES and PROVIDES; given bare names it scores at
     * chance, which is the finding the whole prompt format came from.
     */
    @PostMapping("/matrix/cca-score")
    public ResponseEntity<Map<String, Object>> ccaScore(@RequestBody Map<String, Object> body) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (!onnxCcaService.isLoaded()) {
            out.put("error", "the fine-tuned model is not loaded: "
                    + (onnxCcaService.getLoadError() == null
                       ? "no reason recorded" : onnxCcaService.getLoadError()));
            out.put("hint", "the interface's own analysis does not use this model and is "
                    + "unaffected; see SETUP.md 6.10");
            return ResponseEntity.status(503).body(out);
        }
        Object raw = body.get("pairs");
        if (!(raw instanceof List)) {
            out.put("error", "expected a \"pairs\" array");
            return ResponseEntity.badRequest().body(out);
        }
        List<Double> scores = new ArrayList<>();
        List<Boolean> verdicts = new ArrayList<>();
        for (Object o : (List<?>) raw) {
            Map<?, ?> p = (Map<?, ?>) o;
            double s = onnxCcaService.scorePair(
                    str(p, "pa"), str(p, "a"), str(p, "a_requires"), str(p, "a_provides"),
                    str(p, "pb"), str(p, "b"), str(p, "b_requires"), str(p, "b_provides"));
            scores.add(s);
            verdicts.add(onnxCcaService.isInconsistent(s));
        }
        out.put("scores", scores);
        out.put("inconsistent", verdicts);
        out.put("threshold", onnxCcaService.getThreshold());
        out.put("engine", "onnx-int4-inprocess");
        return ResponseEntity.ok(out);
    }

    private static String str(Map<?, ?> m, String k) {
        Object v = m.get(k);
        return v == null ? "" : String.valueOf(v);
    }

    /** Health and provenance summary, so the interface can say what is actually running. */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> status() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("ccaEngine", ccaEngine);
        Map<String, Object> cca = new LinkedHashMap<>();
        cca.put("loaded", onnxCcaService.isLoaded());
        cca.put("modelDir", onnxCcaService.getModelDir());
        cca.put("threshold", onnxCcaService.getThreshold());
        cca.put("symmetryAveraging", true);
        if (onnxCcaService.getLoadError() != null) {
            cca.put("error", onnxCcaService.getLoadError());
        }
        out.put("fineTunedCca", cca);
        out.put("nliModelLoaded", onnxNliService.isLoaded());
        out.put("embeddingModelLoaded", onnxEmbeddingService.isLoaded());
        out.put("contradictionThreshold", onnxNliService.getContradictionThreshold());
        out.put("overridesEnabled", overrideStore.isEnabled());
        out.put("overrideCount", overrideStore.size());
        out.put("overrideFile", overrideStore.getPath().toString());
        out.put("llm", llmCcaService.status());
        out.put("calibration", calibrationService.status());
        return ResponseEntity.ok(out);
    }

    // --- Analyst overrides: the standing rulings that outrank both models -------------------

    @GetMapping("/overrides")
    public ResponseEntity<Map<String, Object>> listOverrides() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("file", overrideStore.getPath().toString());
        out.put("enabled", overrideStore.isEnabled());
        out.put("count", overrideStore.size());
        out.put("contradictionCount", overrideStore.countByVerdict(OverrideStore.CONTRADICTION));
        out.put("compatibleCount", overrideStore.countByVerdict(OverrideStore.COMPATIBLE));
        out.put("entries", overrideStore.all());
        return ResponseEntity.ok(out);
    }

    /** Records or replaces one ruling. Called on every grid cell the analyst toggles. */
    @PostMapping("/overrides")
    public ResponseEntity<Map<String, Object>> putOverride(@RequestBody OverrideStore.OverrideEntry entry) {
        Map<String, Object> out = new LinkedHashMap<>();
        try {
            overrideStore.put(entry);
            out.put("ok", true);
            out.put("count", overrideStore.size());
            out.put("entry", entry);
        } catch (IllegalArgumentException e) {
            out.put("ok", false);
            out.put("error", e.getMessage());
        }
        return ResponseEntity.ok(out);
    }

    /**
     * Records the whole grid at once.
     *
     * <p>This is what produces a usable training set. Toggling single cells only ever records pairs
     * the model got wrong, which is a biased sample; verifying a grid records the pairs it got right
     * as well, so the calibration head sees both classes.
     */
    @PostMapping("/overrides/bulk")
    public ResponseEntity<Map<String, Object>> putOverridesBulk(
            @RequestBody List<OverrideStore.OverrideEntry> entries) {
        int written = overrideStore.putAll(entries);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("ok", true);
        out.put("written", written);
        out.put("count", overrideStore.size());
        return ResponseEntity.ok(out);
    }

    /** Removes one ruling, so the pair goes back to being decided by the model. */
    @PostMapping("/overrides/delete")
    public ResponseEntity<Map<String, Object>> deleteOverride(@RequestBody OverrideStore.OverrideEntry entry) {
        boolean removed = overrideStore.remove(entry.parameterA, entry.valueA, entry.parameterB, entry.valueB);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("ok", true);
        out.put("removed", removed);
        out.put("count", overrideStore.size());
        return ResponseEntity.ok(out);
    }

    @PostMapping("/overrides/clear")
    public ResponseEntity<Map<String, Object>> clearOverrides() {
        int removed = overrideStore.clear();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("ok", true);
        out.put("removed", removed);
        out.put("count", overrideStore.size());
        return ResponseEntity.ok(out);
    }

    @PostMapping("/overrides/reload")
    public ResponseEntity<Map<String, Object>> reloadOverrides() {
        overrideStore.reload();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("ok", true);
        out.put("count", overrideStore.size());
        return ResponseEntity.ok(out);
    }

    // --- Calibration head ------------------------------------------------------------------

    @GetMapping("/calibration")
    public ResponseEntity<Map<String, Object>> calibrationStatus() {
        return ResponseEntity.ok(calibrationService.status());
    }

    /**
     * Fits the head on every recorded ruling.
     *
     * <p>Slow in proportion to the number of rulings, because each one needs two NLI passes and two
     * embeddings before the fit itself, which is instant.
     */
    @PostMapping("/calibration/train")
    public ResponseEntity<Map<String, Object>> trainCalibration() {
        CalibrationService.TrainResult result = calibrationService.train(overrideStore.trainingSet());
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("ok", result.trained);
        out.put("message", result.message);
        out.put("elapsedMs", result.elapsedMs);
        out.put("status", calibrationService.status());
        return ResponseEntity.ok(out);
    }

    @PostMapping("/calibration/reset")
    public ResponseEntity<Map<String, Object>> resetCalibration() {
        calibrationService.reset();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("ok", true);
        out.put("status", calibrationService.status());
        return ResponseEntity.ok(out);
    }

    // --- PHASE 2: Scenario Intelligence Engine ---
    public static class ScenarioRequest {
        public String scenario;
        public Map<String, String> fixedValues;
        public List<Map<String, String>> solutionSpace;
    }

    @PostMapping("/matrix/match-scenario")
    public ResponseEntity<Map<String, String>> matchScenario(@RequestBody ScenarioRequest request) {
        if (request.solutionSpace == null || request.solutionSpace.isEmpty()) {
            return ResponseEntity.ok(new HashMap<>());
        }

        float[] scenarioVector = onnxEmbeddingService.getEmbedding(request.scenario);
        Map<String, Double> valueScores = new HashMap<>();

        // Score each unique value against the scenario prompt using Cosine Similarity
        for (Map<String, String> config : request.solutionSpace) {
            for (String val : config.values()) {
                if (!valueScores.containsKey(val)) {
                    float[] valVector = onnxEmbeddingService.getEmbedding(val);
                    double score = onnxEmbeddingService.cosineSimilarity(scenarioVector, valVector);
                    valueScores.put(val, score);
                }
            }
        }

        // Filter by fixed values and pick the highest scoring valid configuration
        Map<String, String> bestCombo = null;
        double maxScore = -Double.MAX_VALUE;

        for (Map<String, String> config : request.solutionSpace) {
            // Check fixed value constraints
            boolean matchesFixed = true;
            if (request.fixedValues != null) {
                for (Map.Entry<String, String> fixed : request.fixedValues.entrySet()) {
                    if (fixed.getValue() != null && !fixed.getValue().isEmpty()) {
                        if (!fixed.getValue().equals(config.get(fixed.getKey()))) {
                            matchesFixed = false;
                            break;
                        }
                    }
                }
            }

            if (!matchesFixed) continue;

            double currentScore = 0;
            for (String val : config.values()) {
                currentScore += valueScores.getOrDefault(val, 0.0);
            }

            if (currentScore > maxScore) {
                maxScore = currentScore;
                bestCombo = config;
            }
        }

        return ResponseEntity.ok(bestCombo == null ? new HashMap<>() : bestCombo);
    }

    private void generateValidSolutions(List<String> params, int depth, Map<String, String> current,
                                        Map<String, List<String>> paramMap, Set<String> contradictionPairs,
                                        List<Map<String, String>> solutions) {
        if (depth == params.size()) {
            solutions.add(new LinkedHashMap<>(current));
            return;
        }

        String param = params.get(depth);
        for (String val : paramMap.get(param)) {
            boolean valid = true;
            // Both halves of the key need their parameter, so the already-assigned values are walked
            // as entries rather than values: the same value string under two different parameters is
            // two different cells of the grid and must not share a ruling.
            for (Map.Entry<String, String> assigned : current.entrySet()) {
                if (contradictionPairs.contains(
                        PairKey.of(param, val, assigned.getKey(), assigned.getValue()))) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                current.put(param, val);
                generateValidSolutions(params, depth + 1, current, paramMap, contradictionPairs, solutions);
                current.remove(param);
            }
        }
    }
}
