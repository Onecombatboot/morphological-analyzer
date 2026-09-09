package com.drdo.issa.morphological_analyzer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.SerializationFeature;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A small logistic-regression head over frozen model features, trained on the analyst's own rulings.
 *
 * <p><b>Why this and not a fine-tune.</b> Fine-tuning the NLI network on a few thousand three-word
 * value pairs would mostly teach it to memorise those exact strings, at the cost of the broad
 * linguistic priors that let it work on a domain it has never seen. Since morphological analysis is
 * applied to novel, ill-structured problems, cross-domain behaviour is the one property that cannot
 * be traded away. So both networks stay frozen and only a nine-parameter linear model sits on top.
 *
 * <p><b>What it can and cannot learn.</b> It learns <i>calibration</i>: how much to trust the
 * contradiction score in this domain, how much weight the reverse reading deserves, and where the
 * boundary really sits. It does not learn world knowledge — nothing here can discover that regime
 * change requires a state. That belongs in {@link OverrideStore}, which outranks this.
 *
 * <p><b>Why it is affordable.</b> Nine features and a few hundred labels train in milliseconds on
 * a CPU. There is no GPU, no Python, and no ONNX export anywhere in the loop: the fitted weights are
 * nine doubles written to a JSON file, and scoring is a dot product. It can therefore be retrained
 * in place on the air-gapped machine as more grids are verified.
 *
 * <p><b>Why it is auditable.</b> {@link Coefficient} exposes every weight by name, so the reason a
 * pair was flagged can be stated in a sentence rather than attributed to an opaque model.
 */
@Service
public class CalibrationService {

    /** Feature names, in the order {@link #features} builds them. Persisted with the weights. */
    public static final String[] FEATURE_NAMES = {
            "fwd_contradiction",
            "fwd_entailment",
            "fwd_neutral",
            "rev_contradiction",
            "rev_entailment",
            "rev_neutral",
            "cosine_similarity",
            "max_contradiction",
            "abs_direction_gap"
    };

    public static final int FEATURE_COUNT = FEATURE_NAMES.length;

    /** Below this many labels, or this many of either class, fitting is refused as meaningless. */
    private static final int MIN_SAMPLES = 20;
    private static final int MIN_PER_CLASS = 5;

    private static final int ITERATIONS = 4000;
    private static final double LEARNING_RATE = 0.1;
    private static final double L2 = 0.01;

    @Autowired
    private OnnxNliService onnxNliService;

    @Autowired
    private OnnxEmbeddingService onnxEmbeddingService;

    @Value("${gma.calibration.file:data/calibration.json}")
    private String calibrationFile;

    @Value("${gma.calibration.enabled:true}")
    private boolean enabled;

    private final ObjectMapper mapper = JsonMapper.builder().enable(SerializationFeature.INDENT_OUTPUT).build();

    private volatile Model model;

    // ------------------------------------------------------------------ persisted model

    /** One named weight, reported two ways so the fit can be read without undoing the scaling. */
    public static class Coefficient {
        public String feature;
        /** Weight on the standardised feature. Comparable across features. */
        public double standardised;
        /** Weight per unit of the raw feature. Interpretable in the feature's own units. */
        public double perUnit;

        public Coefficient() {
        }

        public Coefficient(String feature, double standardised, double perUnit) {
            this.feature = feature;
            this.standardised = standardised;
            this.perUnit = perUnit;
        }
    }

    /** The fitted head, exactly as written to disk. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Model {
        public boolean trained;
        public String[] featureNames = FEATURE_NAMES;
        public double[] mean = new double[FEATURE_COUNT];
        public double[] std = new double[FEATURE_COUNT];
        public double[] weights = new double[FEATURE_COUNT];
        public double intercept;
        /** Probability at or above which the head calls a pair contradictory. */
        public double decisionThreshold = 0.5;

        public int samples;
        public int positives;
        public int negatives;
        public String trainedAt;

        /** Accuracy on the data it was fitted to. Optimistic by construction. */
        public double trainAccuracy;
        /**
         * Cross-validated accuracy: the honest number. If this is far below
         * {@link #trainAccuracy} the head has memorised its training pairs and should not be
         * trusted on new ones.
         */
        public double cvAccuracy;
        public int cvFolds;
        /** Share of labels in the majority class: the accuracy of guessing that class every time. */
        public double baselineAccuracy;
        /**
         * How many rulings were recorded one cell at a time rather than as part of a whole grid.
         * Kept for the audit trail only — it is NOT a reliable measure of how much the analyst
         * corrected, because saving a whole grid restamps individually-corrected cells.
         * {@link #disagreements} is the number to trust.
         */
        public int manualLabels;

        /**
         * How many labels contradict what the raw model would have said on its own.
         *
         * <p>This is the honest measure of whether the head can learn anything. A label the model
         * already agrees with teaches it nothing; only the disagreements carry information the
         * model does not already have. Zero disagreements means the head was fitted on the model's
         * own output and every accuracy figure here is circular.
         */
        public int disagreements;

        public List<Coefficient> coefficients = new ArrayList<>();
    }

    // ------------------------------------------------------------------ lifecycle

    @PostConstruct
    public void init() {
        load();
    }

    public boolean isEnabled() {
        return enabled;
    }

    /** True when a fitted head exists and is permitted to take part in grid assessment. */
    public boolean isActive() {
        Model m = model;
        return enabled && m != null && m.trained;
    }

    public Model getModel() {
        return model;
    }

    public Path getPath() {
        return Paths.get(calibrationFile).toAbsolutePath();
    }

    // ------------------------------------------------------------------ features

    /**
     * Builds the feature vector for one pair.
     *
     * <p>Both NLI directions are read, because inference is directional and the disagreement between
     * the two readings is itself informative — a pair that scores high one way and low the other is
     * a different situation from one that scores high both ways, and the head can learn which of
     * those matters in this domain.
     *
     * @param embeddingCache reused across a training run so each distinct value is embedded once
     */
    public double[] features(String parameterA, String valueA,
                             String parameterB, String valueB,
                             Map<String, float[]> embeddingCache) {
        // Put the pair into one fixed orientation first. Training reads rulings in stored order and
        // assessment reads them in declaration order, so without this the head would be fitted on
        // one orientation and applied to the other for about half of all pairs.
        if (!PairKey.aSortsFirst(parameterA, valueA, parameterB, valueB)) {
            String tv = valueA; valueA = valueB; valueB = tv;
        }

        float[] fwd = onnxNliService.rawPair(valueA, valueB);
        float[] rev = onnxNliService.rawPair(valueB, valueA);

        double fc = at(fwd, 0), fe = at(fwd, 1), fn = at(fwd, 2);
        double rc = at(rev, 0), re = at(rev, 1), rn = at(rev, 2);

        float[] va = embed(valueA, embeddingCache);
        float[] vb = embed(valueB, embeddingCache);
        double cos = onnxEmbeddingService.cosineSimilarity(va, vb);

        return new double[]{
                fc, fe, fn,
                rc, re, rn,
                cos,
                Math.max(fc, rc),
                Math.abs(fc - rc)
        };
    }

    private static double at(float[] a, int i) {
        return a != null && a.length > i ? a[i] : 0.0;
    }

    private float[] embed(String text, Map<String, float[]> cache) {
        if (cache == null) return onnxEmbeddingService.getEmbedding(text);
        return cache.computeIfAbsent(text, onnxEmbeddingService::getEmbedding);
    }

    // ------------------------------------------------------------------ scoring

    /**
     * Probability that the pair is contradictory, according to the fitted head.
     *
     * @return a probability in 0..1, or -1 if no head is active
     */
    public double probability(double[] rawFeatures) {
        Model m = model;
        if (m == null || !m.trained || rawFeatures == null || rawFeatures.length != FEATURE_COUNT) {
            return -1;
        }
        double z = m.intercept;
        for (int i = 0; i < FEATURE_COUNT; i++) {
            double s = m.std[i] == 0 ? 1 : m.std[i];
            z += m.weights[i] * ((rawFeatures[i] - m.mean[i]) / s);
        }
        return sigmoid(z);
    }

    public double decisionThreshold() {
        Model m = model;
        return m == null ? 0.5 : m.decisionThreshold;
    }

    private static double sigmoid(double z) {
        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        }
        double e = Math.exp(z);
        return e / (1.0 + e);
    }

    // ------------------------------------------------------------------ training

    /** What a training run produced, returned to the caller and shown in the interface. */
    public static class TrainResult {
        public boolean trained;
        public String message;
        public Model model;
        public long elapsedMs;
    }

    /**
     * Fits the head on every ruling currently in the override store.
     *
     * <p>Each ruling is one label. This is the point of the design: the labelled data is a
     * by-product of ordinary use rather than a dataset someone had to build up front.
     *
     * <p>Feature extraction is the slow part — two NLI passes and two embeddings per pair — so this
     * takes seconds to a minute depending on how many rulings exist. The fit itself is instant.
     */
    public synchronized TrainResult train(List<OverrideStore.OverrideEntry> labels) {
        long started = System.currentTimeMillis();
        TrainResult result = new TrainResult();

        if (labels == null || labels.size() < MIN_SAMPLES) {
            result.trained = false;
            result.message = "Need at least " + MIN_SAMPLES + " recorded rulings to fit anything; have "
                    + (labels == null ? 0 : labels.size())
                    + ". Verify a whole grid rather than toggling single cells: that records the pairs "
                    + "the model got right as well as the ones it got wrong.";
            result.elapsedMs = System.currentTimeMillis() - started;
            return result;
        }

        int positives = 0;
        int manual = 0;
        for (OverrideStore.OverrideEntry e : labels) {
            if (e.isContradiction()) positives++;
            if (OverrideStore.SOURCE_MANUAL.equalsIgnoreCase(e.source)) manual++;
        }
        int negatives = labels.size() - positives;

        if (positives < MIN_PER_CLASS || negatives < MIN_PER_CLASS) {
            result.trained = false;
            result.message = "Need at least " + MIN_PER_CLASS + " rulings of each kind; have "
                    + positives + " contradiction and " + negatives + " compatible. "
                    + "A head fitted on one class only would predict that class for everything.";
            result.elapsedMs = System.currentTimeMillis() - started;
            return result;
        }

        // ---- extract features
        Map<String, float[]> embeddingCache = new HashMap<>();
        int n = labels.size();
        double[][] x = new double[n][];
        double[] y = new double[n];
        float rawThreshold = onnxNliService.getContradictionThreshold();
        int disagreements = 0;
        for (int i = 0; i < n; i++) {
            OverrideStore.OverrideEntry e = labels.get(i);
            x[i] = features(e.parameterA, e.valueA, e.parameterB, e.valueB, embeddingCache);
            y[i] = e.isContradiction() ? 1.0 : 0.0;
            // Feature 0 is the raw contradiction score. Whether the label sits on the other side of
            // the raw threshold is the only honest measure of whether this ruling teaches anything:
            // a label the model already agrees with carries no information the model does not have.
            boolean modelSaysContradiction = x[i][0] > rawThreshold;
            if (modelSaysContradiction != (y[i] == 1.0)) disagreements++;
        }

        // ---- standardise
        double[] mean = new double[FEATURE_COUNT];
        double[] std = new double[FEATURE_COUNT];
        for (int j = 0; j < FEATURE_COUNT; j++) {
            double s = 0;
            for (double[] row : x) s += row[j];
            mean[j] = s / n;
            double v = 0;
            for (double[] row : x) {
                double d = row[j] - mean[j];
                v += d * d;
            }
            std[j] = Math.sqrt(v / n);
            // A feature with no variation carries no information; a std of 1 leaves it at zero
            // after centring rather than producing a division by zero.
            if (std[j] < 1e-9) std[j] = 1.0;
        }
        double[][] z = standardise(x, mean, std);

        // ---- fit
        double[] weights = new double[FEATURE_COUNT];
        double intercept = fit(z, y, weights);

        // ---- honest estimate before reporting anything
        int folds = Math.min(5, Math.min(positives, negatives));
        double cv = folds >= 2 ? crossValidate(x, y, folds) : -1;

        // ---- assemble
        Model m = new Model();
        m.trained = true;
        m.mean = mean;
        m.std = std;
        m.weights = weights;
        m.intercept = intercept;
        m.decisionThreshold = 0.5;
        m.samples = n;
        m.positives = positives;
        m.negatives = negatives;
        m.trainedAt = Instant.now().toString();
        m.cvFolds = folds >= 2 ? folds : 0;
        m.cvAccuracy = cv;
        m.baselineAccuracy = Math.max(positives, negatives) / (double) n;
        m.manualLabels = manual;
        m.disagreements = disagreements;

        int correct = 0;
        for (int i = 0; i < n; i++) {
            double p = sigmoid(dot(weights, z[i]) + intercept);
            if ((p >= m.decisionThreshold ? 1.0 : 0.0) == y[i]) correct++;
        }
        m.trainAccuracy = correct / (double) n;

        m.coefficients = new ArrayList<>();
        for (int j = 0; j < FEATURE_COUNT; j++) {
            m.coefficients.add(new Coefficient(FEATURE_NAMES[j], weights[j], weights[j] / std[j]));
        }

        this.model = m;
        save();

        result.trained = true;
        result.model = m;
        result.elapsedMs = System.currentTimeMillis() - started;
        result.message = describe(m);
        return result;
    }

    private static String describe(Model m) {
        StringBuilder sb = new StringBuilder();
        sb.append("Fitted on ").append(m.samples).append(" rulings (")
                .append(m.positives).append(" contradiction, ").append(m.negatives).append(" compatible). ");
        sb.append(String.format("Training accuracy %.1f%%", m.trainAccuracy * 100));
        if (m.cvFolds >= 2) {
            sb.append(String.format(", %d-fold cross-validated accuracy %.1f%%", m.cvFolds, m.cvAccuracy * 100));
        }
        sb.append(String.format(", always-guess-the-common-class baseline %.1f%%.", m.baselineAccuracy * 100));
        sb.append(String.format(" %d of the %d labels (%.0f%%) disagree with what the raw model would have said"
                + " on its own — those are the ones it can actually learn from.",
                m.disagreements, m.samples, 100.0 * m.disagreements / Math.max(1, m.samples)));
        if (m.disagreements == 0) {
            sb.append(" WARNING: not one label departs from the model's own output, so the head has only learned"
                    + " to reproduce the model it sits on top of and the accuracy above says nothing about whether"
                    + " it is right. Correct the grid where the model is wrong, save it again, and retrain.");
        } else if (m.cvFolds >= 2 && m.cvAccuracy <= m.baselineAccuracy + 0.02) {
            sb.append(" This is no better than guessing — the head is not learning anything useful yet. "
                    + "Verify more grids, or leave calibration off and rely on the raw model plus overrides.");
        } else if (m.cvFolds >= 2 && m.trainAccuracy - m.cvAccuracy > 0.20) {
            sb.append(" Training accuracy is far above the cross-validated figure, so it has partly "
                    + "memorised its training pairs. Treat its output on unseen pairs with caution.");
        }
        return sb.toString();
    }

    private static double[][] standardise(double[][] x, double[] mean, double[] std) {
        double[][] z = new double[x.length][FEATURE_COUNT];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < FEATURE_COUNT; j++) {
                z[i][j] = (x[i][j] - mean[j]) / std[j];
            }
        }
        return z;
    }

    /**
     * Batch gradient descent on the class-weighted log-loss with L2 regularisation.
     *
     * <p>Class weighting matters because most pairs in any real grid are compatible; without it the
     * head would learn to say "compatible" every time and score well doing it.
     *
     * @param weights filled in with the fitted coefficients
     * @return the fitted intercept
     */
    private static double fit(double[][] z, double[] y, double[] weights) {
        int n = z.length;
        double pos = 0;
        for (double v : y) pos += v;
        double neg = n - pos;
        double wPos = pos > 0 ? n / (2.0 * pos) : 1.0;
        double wNeg = neg > 0 ? n / (2.0 * neg) : 1.0;

        double intercept = 0;
        java.util.Arrays.fill(weights, 0.0);

        for (int iter = 0; iter < ITERATIONS; iter++) {
            double[] gradW = new double[FEATURE_COUNT];
            double gradB = 0;
            double totalWeight = 0;

            for (int i = 0; i < n; i++) {
                double w = y[i] == 1.0 ? wPos : wNeg;
                double p = sigmoid(dot(weights, z[i]) + intercept);
                double err = (p - y[i]) * w;
                for (int j = 0; j < FEATURE_COUNT; j++) gradW[j] += err * z[i][j];
                gradB += err;
                totalWeight += w;
            }

            for (int j = 0; j < FEATURE_COUNT; j++) {
                weights[j] -= LEARNING_RATE * (gradW[j] / totalWeight + L2 * weights[j]);
            }
            intercept -= LEARNING_RATE * (gradB / totalWeight);
        }
        return intercept;
    }

    /**
     * Stratified k-fold cross-validation, refitting from scratch on each training split.
     *
     * <p>Standardisation is recomputed inside every fold. Computing it once over the whole set would
     * leak information from the held-out data into the fit and inflate the result.
     */
    private static double crossValidate(double[][] x, double[] y, int folds) {
        int n = x.length;
        int[] fold = new int[n];
        int p = 0, q = 0;
        for (int i = 0; i < n; i++) {
            // Deal the two classes round-robin into folds independently, so every fold holds a
            // similar mix rather than one fold accidentally taking all the contradictions.
            fold[i] = y[i] == 1.0 ? (p++ % folds) : (q++ % folds);
        }

        int correct = 0, counted = 0;
        for (int f = 0; f < folds; f++) {
            List<double[]> trX = new ArrayList<>();
            List<Double> trY = new ArrayList<>();
            List<double[]> teX = new ArrayList<>();
            List<Double> teY = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (fold[i] == f) {
                    teX.add(x[i]);
                    teY.add(y[i]);
                } else {
                    trX.add(x[i]);
                    trY.add(y[i]);
                }
            }
            if (trX.isEmpty() || teX.isEmpty()) continue;

            double[][] trainX = trX.toArray(new double[0][]);
            double[] trainY = new double[trY.size()];
            for (int i = 0; i < trainY.length; i++) trainY[i] = trY.get(i);

            boolean bothClasses = false;
            for (int i = 1; i < trainY.length; i++) {
                if (trainY[i] != trainY[0]) {
                    bothClasses = true;
                    break;
                }
            }
            if (!bothClasses) continue;

            double[] mean = new double[FEATURE_COUNT];
            double[] std = new double[FEATURE_COUNT];
            for (int j = 0; j < FEATURE_COUNT; j++) {
                double s = 0;
                for (double[] row : trainX) s += row[j];
                mean[j] = s / trainX.length;
                double v = 0;
                for (double[] row : trainX) {
                    double d = row[j] - mean[j];
                    v += d * d;
                }
                std[j] = Math.sqrt(v / trainX.length);
                if (std[j] < 1e-9) std[j] = 1.0;
            }

            double[] w = new double[FEATURE_COUNT];
            double b = fit(standardise(trainX, mean, std), trainY, w);

            for (int i = 0; i < teX.size(); i++) {
                double[] raw = teX.get(i);
                double zsum = b;
                for (int j = 0; j < FEATURE_COUNT; j++) {
                    zsum += w[j] * ((raw[j] - mean[j]) / std[j]);
                }
                double pred = sigmoid(zsum) >= 0.5 ? 1.0 : 0.0;
                if (pred == teY.get(i)) correct++;
                counted++;
            }
        }
        return counted == 0 ? -1 : correct / (double) counted;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    // ------------------------------------------------------------------ persistence

    /** Discards the fitted head. The raw model threshold takes over again. */
    public synchronized void reset() {
        this.model = null;
        Path path = getPath();
        try {
            Files.deleteIfExists(path);
        } catch (Exception e) {
            System.err.println("❌ Could not delete " + path + ": " + e.getMessage());
        }
    }

    private synchronized void load() {
        Path path = getPath();
        if (!Files.exists(path)) return;
        try {
            Model m = mapper.readValue(Files.readAllBytes(path), Model.class);
            if (m != null && m.trained
                    && m.weights != null && m.weights.length == FEATURE_COUNT
                    && m.mean != null && m.mean.length == FEATURE_COUNT
                    && m.std != null && m.std.length == FEATURE_COUNT) {
                this.model = m;
                System.out.println("✅ Loaded calibration head fitted on " + m.samples + " ruling(s) from " + path);
            } else {
                System.err.println("❌ Ignoring calibration file with unexpected shape: " + path);
            }
        } catch (Exception e) {
            System.err.println("❌ Could not read calibration from " + path + ": " + e.getMessage());
        }
    }

    private synchronized void save() {
        Path path = getPath();
        try {
            Path parent = path.getParent();
            if (parent != null) Files.createDirectories(parent);
            Path tmp = path.resolveSibling(path.getFileName() + ".tmp");
            Files.write(tmp, mapper.writeValueAsBytes(model));
            try {
                Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicUnsupported) {
                Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (Exception e) {
            System.err.println("❌ Could not save calibration to " + path + ": " + e.getMessage());
        }
    }

    /** Status for the interface, safe to call whether or not a head exists. */
    public Map<String, Object> status() {
        Map<String, Object> out = new LinkedHashMap<>();
        Model m = model;
        out.put("enabled", enabled);
        out.put("trained", m != null && m.trained);
        out.put("active", isActive());
        out.put("file", getPath().toString());
        out.put("featureNames", FEATURE_NAMES);
        out.put("minSamples", MIN_SAMPLES);
        out.put("minPerClass", MIN_PER_CLASS);
        if (m != null && m.trained) {
            out.put("samples", m.samples);
            out.put("positives", m.positives);
            out.put("negatives", m.negatives);
            out.put("manualLabels", m.manualLabels);
            out.put("disagreements", m.disagreements);
            out.put("trainedAt", m.trainedAt);
            out.put("trainAccuracy", m.trainAccuracy);
            out.put("cvAccuracy", m.cvAccuracy);
            out.put("cvFolds", m.cvFolds);
            out.put("baselineAccuracy", m.baselineAccuracy);
            out.put("decisionThreshold", m.decisionThreshold);
            out.put("intercept", m.intercept);
            out.put("coefficients", m.coefficients);
            out.put("summary", describe(m));
        }
        return out;
    }
}
