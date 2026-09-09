package com.drdo.issa.morphological_analyzer;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Map;

/**
 * Decides whether one value pair is consistent, and says on whose authority.
 *
 * <p>Three signals are consulted in a fixed order of precedence:
 *
 * <ol>
 *   <li><b>The analyst's standing ruling</b> ({@link OverrideStore}), if one exists. It always wins.
 *       Cross-consistency assessment is expert judgement in Zwicky's and Ritchey's method; a
 *       recorded judgement must never be quietly overturned by a model.
 *   <li><b>The calibration head</b> ({@link CalibrationService}), if one has been fitted and is
 *       enabled. It reweights the raw model signals using what the analyst has taught it.
 *   <li><b>The raw NLI model</b> ({@link OnnxNliService}), thresholded. The fallback, and what the
 *       system did before any of this existed.
 * </ol>
 *
 * <p>Every verdict carries its provenance, so the grid can show which cells are the analyst's own
 * and which are the machine's guess. That distinction is the difference between a result a reviewer
 * can interrogate and one they have to take on trust.
 */
@Service
public class ConsistencyService {

    public static final String SOURCE_OVERRIDE = "OVERRIDE";
    public static final String SOURCE_CALIBRATED = "CALIBRATED";
    public static final String SOURCE_MODEL = "MODEL";
    public static final String SOURCE_LLM = "LLM";
    /** Verdict from the QLoRA model fine-tuned on the cross-consistency corpus. */
    public static final String SOURCE_FINETUNED = "FINETUNED";

    @Autowired
    private OnnxNliService onnxNliService;

    @Autowired
    private OverrideStore overrideStore;

    @Autowired
    private CalibrationService calibrationService;

    @Autowired
    private LlmCcaService llmCcaService;

    /** One decision, with the reasoning attached. */
    public static class Verdict {
        public boolean contradiction;
        /** {@link #SOURCE_OVERRIDE}, {@link #SOURCE_CALIBRATED} or {@link #SOURCE_MODEL}. */
        public String source;
        /** The number the decision was made on: a probability, or 1/0 for a standing ruling. */
        public double score;
        /** A sentence a reviewer can read. */
        public String detail;

        Verdict(boolean contradiction, String source, double score, String detail) {
            this.contradiction = contradiction;
            this.source = source;
            this.score = score;
            this.detail = detail;
        }
    }

    /**
     * Assesses a pair.
     *
     * <p>The pair must already be in parameter-declaration order: NLI is directional, and a single
     * correctly oriented reading measured better than any way of combining both readings. The
     * ordering is the caller's job because only the caller knows the declaration order.
     *
     * @param embeddingCache optional, reused across a grid so each distinct value is embedded once
     */
    public Verdict assess(String parameterA, String valueA,
                          String parameterB, String valueB,
                          Map<String, float[]> embeddingCache) {
        return assess(parameterA, valueA, parameterB, valueB, embeddingCache, null);
    }

    /**
     * @param llmVerdicts verdicts produced for the whole matrix in one pass, or null if none.
     *                    Passed in rather than fetched here because the model is asked once per
     *                    parameter pair, not once per cell -- that batching is what makes the
     *                    judgements consistent with each other.
     */
    public Verdict assess(String parameterA, String valueA,
                          String parameterB, String valueB,
                          Map<String, float[]> embeddingCache,
                          Map<String, LlmCcaService.PairVerdict> llmVerdicts) {

        OverrideStore.OverrideEntry ruling = overrideStore.lookup(parameterA, valueA, parameterB, valueB);
        if (ruling != null) {
            boolean x = ruling.isContradiction();
            String note = ruling.note == null || ruling.note.isBlank() ? "" : " — " + ruling.note;
            return new Verdict(x, SOURCE_OVERRIDE, x ? 1.0 : 0.0,
                    "Analyst ruling recorded " + ruling.recordedAt + note);
        }

        if (llmVerdicts != null) {
            LlmCcaService.PairVerdict pv = llmVerdicts.get(PairKey.of(parameterA, valueA, parameterB, valueB));
            if (pv != null) {
                // The score carried through is the model's own P(inconsistent), not a flattened
                // 1/0, so the grid can shade by confidence and the analyst sees which exclusions
                // the model was actually sure about.
                return new Verdict(pv.contradiction, SOURCE_LLM, pv.confidence, pv.detail);
            }
        }

        if (calibrationService.isActive()) {
            double[] f = calibrationService.features(parameterA, valueA, parameterB, valueB, embeddingCache);
            double p = calibrationService.probability(f);
            if (p >= 0) {
                double t = calibrationService.decisionThreshold();
                return new Verdict(p >= t, SOURCE_CALIBRATED, p,
                        String.format("Calibration head scored %.3f against a threshold of %.2f", p, t));
            }
        }

        double raw = onnxNliService.contradictionScore(valueA, valueB);
        double t = onnxNliService.getContradictionThreshold();
        return new Verdict(raw > t, SOURCE_MODEL, raw,
                String.format("NLI contradiction score %.3f against a threshold of %.2f", raw, t));
    }
}
