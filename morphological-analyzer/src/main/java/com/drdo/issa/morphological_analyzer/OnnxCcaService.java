package com.drdo.issa.morphological_analyzer;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * The fine-tuned cross-consistency model, run in-process on the CPU.
 *
 * <p>This replaces {@code serve_cca.py}. It is the same model and the same
 * arithmetic, with no Python, no HTTP hop and no second process: a QLoRA
 * adapter on Qwen2.5-3B, merged, exported to ONNX and quantised to 4-bit.
 *
 * <h2>How a verdict is produced</h2>
 * The model was trained so that the token immediately after {@code "VERDICT:"}
 * is either {@code Y} or {@code N}. Scoring therefore needs no text generation
 * at all: one forward pass, read the logits at the final position, and take a
 * softmax over just those two token ids. That is the whole computation, and it
 * is what makes the operating point an explicit number rather than something
 * that emerges from decoding.
 *
 * <p>{@code P(inconsistent)} is the N side of that pair -- "no, one scenario
 * cannot contain both".
 *
 * <h2>Why both presentation orders</h2>
 * Cross-consistency is symmetric: consistent(A,B) is the same claim as
 * consistent(B,A). Each cell is scored both ways and the probabilities
 * averaged. This is not a refinement that can be dropped for speed. Measured on
 * the held-out set, dropping it takes F1 at the deployed operating point from
 * 0.805 to 0.686 for the reference model, and collapses the 4-bit model's
 * selective-prediction curve to zero -- the confidence signal inverts. It costs
 * a second forward pass and it is load-bearing.
 *
 * <h2>The prompt is not editable</h2>
 * The wording below is byte-identical to {@code build_dataset.SYSTEM_CELL} and
 * {@code serve_cca._one()}, including the em dash in "Facet A —". The model was
 * trained on this exact text. Any drift degrades it silently: the verdicts stay
 * fluent and plausible, and the numbers quietly stop meaning what they used to.
 * If the training format ever changes, change it here in the same commit.
 *
 * <h2>CPU only</h2>
 * The session is created with the default CPU provider and no execution
 * provider is registered. There is no GPU path to fall back from and none is
 * wanted.
 */
@Service
public class OnnxCcaService {

    /** Exactly build_dataset.SYSTEM_CELL. Do not reword. */
    private static final String SYSTEM =
            "You are a defence and security analyst carrying out cross-consistency assessment "
            + "for General Morphological Analysis. You are given two options from different "
            + "facets of ONE scenario, with what each requires and provides. Decide whether a "
            + "single real scenario could contain both. Answer N only when you can name the "
            + "clause that makes it impossible or means it never occurs; ineffective or unwise "
            + "is still Y.";

    /** calibrate.PREFIX. The verdict token is the one immediately after this. */
    private static final String PREFIX = "VERDICT:";

    @Value("${gma.onnxcca.enabled:true}")
    private boolean enabled;

    @Value("${gma.onnxcca.model-dir:models/cca-int4}")
    private String modelDir;

    /** Chosen leave-one-domain-out on the held-out set. */
    @Value("${gma.onnxcca.threshold:0.5}")
    private double threshold;

    /** 0 lets ONNX Runtime decide. Pin it to the physical core count. */
    @Value("${gma.onnxcca.threads:0}")
    private int threads;

    private OrtEnvironment env;
    private OrtSession session;
    private HuggingFaceTokenizer tokenizer;
    private long yId = -1;
    private long nId = -1;
    private volatile boolean loaded = false;
    private String loadError = null;

    @PostConstruct
    public void init() {
        if (!enabled) {
            loadError = "disabled by gma.onnxcca.enabled=false";
            System.out.println("i ONNX CCA model disabled (gma.onnxcca.enabled=false); "
                    + "the interface is unaffected, it uses llama.cpp");
            return;
        }
        try {
            Path dir = Paths.get(modelDir).toAbsolutePath();
            File model = dir.resolve("model.onnx").toFile();
            if (!model.exists()) {
                loadError = "model.onnx not found under " + dir;
                System.err.println("x ONNX CCA model: " + loadError);
                return;
            }

            tokenizer = HuggingFaceTokenizer.newInstance(dir.resolve("tokenizer.json"));

            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            if (threads > 0) {
                opts.setIntraOpNumThreads(threads);
            }
            session = env.createSession(model.getAbsolutePath(), opts);

            resolveVerdictTokens();

            loaded = true;
            System.out.println("+ ONNX CCA model loaded (CPU): " + dir.getFileName()
                    + "  verdict ids Y=" + yId + " N=" + nId
                    + "  threshold=" + threshold
                    + "  threads=" + (threads > 0 ? String.valueOf(threads) : "auto"));
        } catch (Throwable t) {
            loadError = t.getMessage();
            System.err.println("x ONNX CCA model failed to load: " + t);
        }
    }

    /**
     * Finds the ids of the Y and N that follow the prefix.
     *
     * <p>Derived from the tokenizer rather than hard-coded, and then checked:
     * the verdict must be exactly one token past the prefix. If it is not, the
     * logits are being read at the wrong position and every probability the
     * service returns would be meaningless, so this refuses to load instead.
     */
    private void resolveVerdictTokens() {
        long[] pre = tokenizer.encode(PREFIX, false, false).getIds();
        long[] withY = tokenizer.encode(PREFIX + " Y", false, false).getIds();
        long[] withN = tokenizer.encode(PREFIX + " N", false, false).getIds();

        if (withY.length != pre.length + 1 || withN.length != pre.length + 1) {
            throw new IllegalStateException(
                    "the verdict is not a single token after \"" + PREFIX + "\" ("
                    + pre.length + " -> " + withY.length + "/" + withN.length
                    + "); the logit position would be wrong");
        }
        yId = withY[withY.length - 1];
        nId = withN[withN.length - 1];
        if (yId == nId) {
            throw new IllegalStateException("Y and N tokenise identically; cannot score by logit");
        }
    }

    /** The user turn. Byte-identical to serve_cca._one(). */
    private String userTurn(String pa, String a, String aReq, String aProv,
                            String pb, String b, String bReq, String bProv) {
        return "Facet A — " + pa + "\n"
             + a + " | REQUIRES: " + aReq + " | PROVIDES: " + aProv + "\n"
             + "\n"
             + "Facet B — " + pb + "\n"
             + b + " | REQUIRES: " + bReq + " | PROVIDES: " + bProv + "\n"
             + "\n"
             + "Can one scenario contain both?";
    }

    /** Qwen's ChatML template with add_generation_prompt=True, then the prefix. */
    private String buildPrompt(String user) {
        return "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
             + "<|im_start|>user\n" + user + "<|im_end|>\n"
             + "<|im_start|>assistant\n"
             + PREFIX;
    }

    /** P(inconsistent) for one presentation order. */
    private double scoreOne(String prompt) throws Exception {
        Encoding enc = tokenizer.encode(prompt, false, false);
        long[] ids = enc.getIds();
        int n = ids.length;

        long[] mask = new long[n];
        long[] pos = new long[n];
        for (int i = 0; i < n; i++) {
            mask[i] = 1L;
            pos[i] = i;
        }

        long[] shape = {1, n};
        Map<String, OnnxTensor> in = new HashMap<>();
        try {
            in.put("input_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape));
            in.put("attention_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape));
            in.put("position_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(pos), shape));

            try (OrtSession.Result out = session.run(in)) {
                OnnxTensor logits = (OnnxTensor) out.get(0);
                // The graph emits logits for every position: [1, n, vocab]. At a
                // 350-token prompt and a 151936 vocab that is over 200 MB, so it
                // is read straight from the native buffer. Calling getValue()
                // here would allocate a float[1][n][151936] per cell and bury
                // the collector.
                FloatBuffer buf = logits.getFloatBuffer();
                long[] sh = logits.getInfo().getShape();
                int width = (int) sh[sh.length - 1];

                float ly, ln;
                if (width == 2) {
                    // Trimmed head. onnx_trim_head.py sliced the output
                    // projection to the two verdict columns and the final
                    // position, so the whole output is {Y, N}. Same numbers as
                    // the full head, without computing 151934 logits per token
                    // and shipping 200 MB across JNI to discard it.
                    ly = buf.get(0);
                    ln = buf.get(1);
                } else {
                    // Untrimmed export: [1, seq, vocab], read the last position.
                    int base = (n - 1) * width;
                    ly = buf.get(base + (int) yId);
                    ln = buf.get(base + (int) nId);
                }

                // softmax over just the two verdict logits
                double m = Math.max(ly, ln);
                double ey = Math.exp(ly - m);
                double en = Math.exp(ln - m);
                return en / (ey + en);
            }
        } finally {
            for (OnnxTensor t : in.values()) {
                t.close();
            }
        }
    }

    /**
     * P(inconsistent) for a cell, averaged over both presentation orders.
     *
     * @return the probability, or -1 if the model is not loaded.
     */
    public double scorePair(String pa, String a, String aReq, String aProv,
                            String pb, String b, String bReq, String bProv) {
        if (!loaded) {
            return -1;
        }
        try {
            double fwd = scoreOne(buildPrompt(userTurn(pa, a, aReq, aProv, pb, b, bReq, bProv)));
            double rev = scoreOne(buildPrompt(userTurn(pb, b, bReq, bProv, pa, a, aReq, aProv)));
            return 0.5 * (fwd + rev);
        } catch (Exception e) {
            System.err.println("ONNX CCA scoring failed: " + e.getMessage());
            return -1;
        }
    }

    /** Single order only. Faster, and materially worse -- see the class comment. */
    public double scorePairSingleOrder(String pa, String a, String aReq, String aProv,
                                       String pb, String b, String bReq, String bProv) {
        if (!loaded) {
            return -1;
        }
        try {
            return scoreOne(buildPrompt(userTurn(pa, a, aReq, aProv, pb, b, bReq, bProv)));
        } catch (Exception e) {
            System.err.println("ONNX CCA scoring failed: " + e.getMessage());
            return -1;
        }
    }

    public boolean isInconsistent(double p) {
        return p >= threshold;
    }

    public boolean isLoaded() {
        return loaded;
    }

    public double getThreshold() {
        return threshold;
    }

    public String getModelDir() {
        return modelDir;
    }

    public String getLoadError() {
        return loadError;
    }

    @PreDestroy
    public void close() {
        try {
            if (session != null) session.close();
        } catch (Exception ignored) {
            // shutting down anyway
        }
        if (tokenizer != null) tokenizer.close();
    }
}
