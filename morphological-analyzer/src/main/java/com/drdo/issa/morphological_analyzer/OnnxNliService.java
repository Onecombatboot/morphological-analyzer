package com.drdo.issa.morphological_analyzer;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.huggingface.translator.TextClassificationTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.StringPair;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Paths;

@Service
public class OnnxNliService {

    private ZooModel<StringPair, float[]> model;

    /**
     * Measured on a labelled set, contradictory pairs scored 0.89-1.00 and compatible pairs
     * 0.06-0.56, so the boundary sits in the empty band between them. The default here matches
     * application.properties; an earlier 0.72 was left over from calibration on the threat domain
     * and was lowered when the tool was tried on unrelated ones.
     */
    @Value("${gma.nli.contradiction-threshold:0.70}")
    private float contradictionThreshold;

    @Value("${gma.nli.model-dir:models/nli-deberta-v3-large}")
    private String modelDir;

    /** Optional. Leave unset to use the machine's own temp directory. */
    @Value("${gma.nli.temp-dir:}")
    private String tempDir;

    /**
     * The NLI model is no longer the cross-consistency engine — it measured AUC 0.53 against expert
     * labels, which is a coin toss — and it is not shipped in the offline package. Loading is
     * skipped by default so its absence is stated once, cleanly, rather than reported as a failure
     * every time the application starts.
     */
    @Value("${gma.nli.enabled:false}")
    private boolean enabled;

    @PostConstruct
    public void init() {
        if (!enabled) {
            System.out.println("ℹ NLI scoring disabled (gma.nli.enabled=false); "
                    + "cross-consistency is handled by the local instruction model");
            return;
        }
        try {
            // DJL unpacks native libraries into the temp directory. A path is only forced here if
            // one was configured; otherwise the machine's own temp directory is used, so nothing
            // depends on a particular folder existing on a particular drive.
            if (tempDir != null && !tempDir.isBlank()) {
                java.io.File dir = new java.io.File(tempDir);
                if (dir.exists() || dir.mkdirs()) {
                    System.setProperty("java.io.tmpdir", dir.getAbsolutePath());
                }
            }

            String modelPath = Paths.get(modelDir).toAbsolutePath().toString();

            Criteria<StringPair, float[]> criteria = Criteria.builder()
                    .setTypes(StringPair.class, float[].class)
                    .optModelPath(Paths.get(modelPath))
                    .optEngine("OnnxRuntime")
                    .optDevice(Device.cpu()) // 🛡️ Forces CPU execution, blocking DLL GPU crashes
                    .optArgument("reranking", "true")
                    .optTranslatorFactory(new TextClassificationTranslatorFactory())
                    .build();

            this.model = criteria.loadModel();
            System.out.println("✅ ONNX NLI Model loadded successfully natively in Java!");
        } catch (IOException | ModelException e) {
            System.err.println("❌ Failed to load ONNX NLI model: " + e.getMessage());
        }
    }

    /** Runs an arbitrary premise/hypothesis pair, for comparing candidate framings. */
    public float[] rawPair(String premise, String hypothesis) {
        if (model == null) return new float[0];
        try (Predictor<StringPair, float[]> predictor = model.newPredictor()) {
            return predictor.predict(new StringPair(premise, hypothesis));
        } catch (Exception e) {
            return new float[0];
        }
    }

    /**
     * Scores how strongly the second text contradicts the first, in 0..1.
     *
     * <p>This is class 0 of the NLI head. The class order is [contradiction, entailment, neutral],
     * taken from the model's own {@code config.json} {@code id2label} map — check that field before
     * trusting index 0 if the model is ever swapped, because label order is not standardised across
     * NLI checkpoints and a mismatch inverts every result silently rather than failing.
     *
     * <p>Both texts are passed exactly as given, with no wrapper sentence. Four framings were
     * compared across two unrelated domains and bare values were the only one that held up in both:
     * anything naming an actor scored best on threat scenarios and was meaningless on civil-defence
     * planning, which has no actor.
     *
     * <p>The reading is directional, so callers must pass the pair in the order the parameters were
     * declared. Scoring both directions and combining them measured worse than one correctly
     * oriented reading, whichever way the two were combined.
     */
    public float contradictionScore(String textA, String textB) {
        if (model == null) return 0f;
        try (Predictor<StringPair, float[]> predictor = model.newPredictor()) {
            float[] logits = predictor.predict(new StringPair(textA, textB));
            return logits.length > 0 ? logits[0] : 0f;
        } catch (Exception e) {
            System.err.println("Error running ONNX prediction: " + e.getMessage());
            return 0f;
        }
    }

    public boolean isContradiction(String textA, String textB) {
        return contradictionScore(textA, textB) > contradictionThreshold;
    }

    /** The configured threshold, so callers reporting a decision can state what it was judged against. */
    public float getContradictionThreshold() {
        return contradictionThreshold;
    }

    /** True if the model loaded. When false every score is 0 and nothing is ever marked. */
    public boolean isLoaded() {
        return model != null;
    }
}