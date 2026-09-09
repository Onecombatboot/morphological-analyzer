package com.drdo.issa.morphological_analyzer;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Cross-consistency assessment by a local instruction-tuned model.
 *
 * <p>Three things distinguish this from asking the model cell by cell, and each exists because the
 * simpler version was measured and failed.
 *
 * <p><b>1. It judges a whole parameter pair at once.</b> The first version asked about one pair per
 * call. Nothing tied those calls together, so the same concept got opposite verdicts on trivial
 * wording differences, and a matrix cost 176 generations. Showing every value of both parameters at
 * once forces the model to rank them against each other and collapses 176 calls into
 * {@code n(n-1)/2} — six for four parameters.
 *
 * <p><b>2. It writes a domain brief before judging anything.</b> Shown two bare noun phrases and
 * required to answer on the next token, the model excluded ransomware against every attacker type
 * and rail disruption against every trigger — knowledge it demonstrably has but never activated.
 * The brief makes it first say what each value implies, and what question connects each pair of
 * dimensions, so the grid calls compare descriptions rather than tokens. The glosses are the model's
 * own; nothing here is domain-specific or supplied by hand.
 *
 * <p><b>3. It reads the confidence behind each verdict.</b> The grammar restricts every verdict
 * token to {@code Y} or {@code N}, so the probability mass sits on two tokens and a real
 * P(inconsistent) is available at no extra cost. A confident exclusion sits near 1.0; a guess sits
 * near 0.5. Thresholding on that filters out exactly the unconvinced exclusions that empty the
 * solution space.
 *
 * <p><b>Why the grid itself stays grammar-constrained.</b> Letting the model write freely was
 * measured at <b>515 seconds</b> for one matrix against 45, and the loose parse still recovered only
 * 128 of 176 cells. The grammar pins the answer shape so nothing can drift; the thinking happens in
 * the brief instead, where it is paid for once rather than six times.
 *
 * <p><b>Determinism.</b> Temperature 0 and a fixed seed, so the same matrix always yields the same
 * verdicts — a result that changes between runs cannot be defended to a reviewer. This holds only
 * while the thread count is fixed too, since that changes the order of floating-point reductions.
 */
@Service
public class LlmCcaService {

    private LlamaModel model;

    @Value("${gma.llm.enabled:true}")
    private boolean enabled;

    @Value("${gma.llm.model:models/Qwen2.5-7B-Instruct-Q4_K_M.gguf}")
    private String modelPath;

    @Value("${gma.llm.ctx-size:4096}")
    private int ctxSize;

    @Value("${gma.llm.threads:0}")
    private int threads;

    @Value("${gma.llm.criterion:}")
    private String criterionOverride;

    @Value("${gma.llm.suppress-thinking:false}")
    private boolean suppressThinking;

    /** Which chat template to wrap the prompt in: chatml for the Qwen family, gemma for Gemma 3. */
    @Value("${gma.llm.template:chatml}")
    private String template;

    /** Write a domain brief before judging. Off restores the previous single-pass behaviour. */
    @Value("${gma.llm.brief:true}")
    private boolean useBrief;

    /** Cap on the brief. Every token here is paid for once, but on CPU it is still real time. */
    @Value("${gma.llm.brief-max-tokens:160}")
    private int briefMaxTokens;

    /**
     * Ask the brief for a one-line gloss of every value as well as the relationships.
     *
     * <p>OFF, on measurement. Including glosses scored 43.2% raw / 50.7% balanced against 176 expert
     * labels; dropping them took the same build to 55.7% / 58.1%. They also tripled grid-call time,
     * because a gloss line per value is carried into all six prompts and prefill runs at roughly 20
     * tokens a second. The relationship lines are short, are reused the same way, and are the half
     * that actually helped.
     */
    @Value("${gma.llm.brief-glosses:false}")
    private boolean briefGlosses;

    /**
     * A pair is excluded only when P(inconsistent) reaches this. 0.5 reproduces the plain verdict.
     * Raising it keeps only confident exclusions, which is the direct lever against over-pruning.
     */
    @Value("${gma.llm.confidence-threshold:0.5}")
    private double confidenceThreshold;

    /**
     * Read per-token probabilities to score each verdict instead of taking a bare Y/N.
     *
     * <p>OFF, because it crashes the bundled runtime. {@code setNProbs} exists on the Java API and
     * {@code LlamaOutput.probabilities} is populated in general, but with a GBNF grammar active
     * llama.cpp build 4916 aborts inside its own JSON layer --
     * {@code GGML_ASSERT(it != m_data.m_value.object->end())} -- because the probability field it
     * looks up is never written for grammar-constrained sampling. The plumbing is kept so this can
     * be switched on if the inference layer is ever replaced.
     */
    @Value("${gma.llm.confidence-scores:false}")
    private boolean confidenceScores;

    /**
     * The default exclusion bar. Overridable with gma.llm.criterion.
     *
     * <p>Chosen by measurement. Five wordings were run over a matrix carrying 176 expert labels; the
     * count excluded ranged from 60 to 126 on this sentence alone, against an expert count of 119.
     * Framing the bar as impossibility scored 49.2% balanced accuracy — below chance. Framing it as
     * "would you expect to see this in a real case" scored 60.9%. Cross-consistency is a question
     * about typicality, not physics.
     */
    static final String DEFAULT_CRITERION =
            "Mark Y only if this is a combination you would expect to see in a real documented case. "
          + "If it would be unusual, contrived, or a stretch, mark N.";

    private static final String SYSTEM_PREFIX =
            "You are a defence and security analyst carrying out cross-consistency assessment for "
          + "General Morphological Analysis. Two lists describe two facets of ONE single scenario. "
          + "For each combination you decide whether that one scenario could plausibly have both "
          + "facets at once. ";

    @PostConstruct
    public void init() {
        if (!enabled) {
            System.out.println("\u2139 LLM cross-consistency disabled (gma.llm.enabled=false)");
            return;
        }
        try {
            String abs = Paths.get(modelPath).toAbsolutePath().toString();
            if (!Files.exists(Paths.get(abs))) {
                System.err.println("\u274c LLM model not found at " + abs);
                return;
            }
            ModelParameters mp = new ModelParameters()
                    // CPU only. The deployment machine has no usable GPU, and letting the loader
                    // probe for one crashes natively rather than throwing something catchable.
                    .setGpuLayers(0)
                    .setCtxSize(ctxSize)
                    .setModel(abs);
            if (threads > 0) mp = mp.setThreads(threads);

            long t = System.currentTimeMillis();
            model = new LlamaModel(mp);
            System.out.printf("\u2705 LLM loaded for cross-consistency in %.1fs: %s%n",
                    (System.currentTimeMillis() - t) / 1000.0, abs);
        } catch (Exception e) {
            // A missing or broken model must not stop the application: the analyst's standing
            // rulings still apply, and starting with an empty grid beats not starting at all.
            System.err.println("\u274c Failed to load LLM: " + e.getMessage());
            model = null;
        }
    }

    @PreDestroy
    public void close() {
        if (model != null) {
            try {
                model.close();
            } catch (Exception ignored) {
                // Nothing useful to do while shutting down.
            }
        }
    }

    public boolean isLoaded() {
        return model != null;
    }

    public String criterion() {
        return criterionOverride == null || criterionOverride.isBlank()
                ? DEFAULT_CRITERION : criterionOverride;
    }

    /** One assessed pair, with the confidence behind it and the text a reviewer can read. */
    /** Factory so another engine can produce verdicts in this same shape. */
    static PairVerdict newPairVerdict(boolean contradiction, double confidence, String detail) {
        return new PairVerdict(contradiction, confidence, detail);
    }

    public static class PairVerdict {
        public final boolean contradiction;
        /** P(inconsistent) in 0..1 as the model scored it. */
        public final double confidence;
        public final String detail;

        PairVerdict(boolean contradiction, double confidence, String detail) {
            this.contradiction = contradiction;
            this.confidence = confidence;
            this.detail = detail;
        }
    }

    // ------------------------------------------------------------------ the domain brief

    /** What the model worked out about the matrix before judging any pair of it. */
    static class Brief {
        String glosses = "";
        /** Keyed by the two parameter names sorted and joined, so lookup is order-independent. */
        final Map<String, String> relationships = new HashMap<>();

        static String key(String a, String b) {
            return a.compareTo(b) <= 0 ? a + "\u0000" + b : b + "\u0000" + a;
        }
    }

    /**
     * Asks the model to say what each value implies, and what question connects each pair of
     * dimensions, before any judging happens.
     *
     * <p>The relationship half matters as much as the glosses. Measured per parameter pair on a
     * labelled matrix, the model scored 75% where the question was one of scale coherence (MEANS
     * against METHOD) and 50% — chance — where it was one of capability (GOAL against MEANS). Those
     * are different questions and the previous prompt asked both the same generic way. Having the
     * model name the relationship itself, from the parameter names alone, keeps that automatic.
     */
    Brief buildBrief(List<String> parameters, Map<String, List<String>> values) {
        Brief brief = new Brief();
        if (model == null) return brief;

        StringBuilder u = new StringBuilder();
        u.append("A scenario is described by these dimensions and options:\n\n");
        for (String p : parameters) {
            u.append(p).append(": ").append(String.join(", ", sanitise(values.get(p)))).append("\n");
        }
        u.append("\nWrite two sections and nothing else.\n\n");
        // Relationships first on purpose: this section is short and it is the half that was
        // lost when the brief hit its token cap partway through the glosses.
        u.append("RELATIONSHIPS\n");
        u.append("One line per pair of dimensions, in the form  DIM_A ~ DIM_B: the question "
               + "that decides whether an option from each can coexist. One short question each.\n");
        u.append("Pairs to cover:\n");
        for (int i = 0; i < parameters.size(); i++) {
            for (int j = i + 1; j < parameters.size(); j++) {
                u.append("  ").append(parameters.get(i)).append(" ~ ")
                 .append(parameters.get(j)).append("\n");
            }
        }
        if (briefGlosses) {
            u.append("\nGLOSSES\n");
            u.append("One line per option, in the form  Option: what it requires or implies.\n");
            u.append("Name the actor type, scale, resources or preconditions it assumes. "
                   + "Four words maximum.\n");
        }
        u.append("RELATIONSHIPS\n");
        u.append("One line per pair of dimensions, in the form  DIM_A ~ DIM_B: the question that "
               + "decides whether an option from each can coexist.\n");
        u.append("Pairs to cover:\n");
        for (int i = 0; i < parameters.size(); i++) {
            for (int j = i + 1; j < parameters.size(); j++) {
                u.append("  ").append(parameters.get(i)).append(" ~ ")
                 .append(parameters.get(j)).append("\n");
            }
        }

        String system = "You are a defence and security analyst preparing a morphological analysis. "
                      + "You are precise and extremely brief.";

        InferenceParameters ip = new InferenceParameters(buildPrompt(system, u.toString()))
                .setTemperature(0.0f)
                .setSeed(42)
                .setCachePrompt(true)
                .setNPredict(briefMaxTokens);

        String raw;
        try {
            raw = model.complete(ip).trim();
        } catch (Exception e) {
            System.err.println("\u274c Domain brief failed, judging without it: " + e.getMessage());
            return brief;
        }

        StringBuilder gl = new StringBuilder();
        boolean inRel = false;
        for (String line : raw.split("\\R")) {
            String t = line.trim();
            if (t.equalsIgnoreCase("RELATIONSHIPS")) { inRel = true; continue; }
            if (t.equalsIgnoreCase("GLOSSES")) { inRel = false; continue; }
            if (t.isEmpty()) continue;
            if (!inRel) {
                gl.append(t).append("\n");
                continue;
            }
            int tilde = t.indexOf('~');
            if (tilde <= 0) continue;
            int colon = t.indexOf(':', tilde);
            if (colon <= tilde) continue;
            String a = t.substring(0, tilde).trim().replaceAll("^[-*\\s]+", "");
            String b = t.substring(tilde + 1, colon).trim();
            String q = t.substring(colon + 1).trim();
            for (String pa : parameters) {
                for (String pb : parameters) {
                    if (pa.equalsIgnoreCase(a) && pb.equalsIgnoreCase(b)) {
                        brief.relationships.put(Brief.key(pa, pb), q);
                    }
                }
            }
        }
        brief.glosses = gl.toString().trim();
        return brief;
    }

    // ------------------------------------------------------------------ assessment

    /**
     * Assesses every cross-parameter pair in the matrix.
     *
     * @return verdicts keyed by {@link PairKey}; empty if the model is unavailable, in which case
     *         the caller marks nothing rather than guessing
     */
    public Map<String, PairVerdict> assessMatrix(List<String> parameters,
                                                 Map<String, List<String>> values) {
        Map<String, PairVerdict> out = new HashMap<>();
        if (model == null) return out;

        long t0 = System.currentTimeMillis();
        Brief brief = useBrief ? buildBrief(parameters, values) : new Brief();
        if (useBrief) {
            System.out.printf("\u2139 domain brief written in %.1fs (%d gloss lines, %d relationships)%n",
                    (System.currentTimeMillis() - t0) / 1000.0,
                    brief.glosses.isEmpty() ? 0 : brief.glosses.split("\\R").length,
                    brief.relationships.size());
        }

        String rule = criterion();
        String system = SYSTEM_PREFIX + rule;

        for (int i = 0; i < parameters.size(); i++) {
            for (int j = i + 1; j < parameters.size(); j++) {
                String pa = parameters.get(i);
                String pb = parameters.get(j);
                List<String> va = sanitise(values.get(pa));
                List<String> vb = sanitise(values.get(pb));
                if (va.isEmpty() || vb.isEmpty()) continue;

                try {
                    assessPair(system, rule, brief, pa, va, pb, vb, out);
                } catch (Exception e) {
                    // One failed parameter pair leaves those cells unmarked rather than aborting
                    // the whole analysis; the analyst can still rule on them by hand.
                    System.err.println("\u274c LLM assessment failed for " + pa + " x " + pb
                            + ": " + e.getMessage());
                }
            }
        }
        return out;
    }

    private void assessPair(String system, String rule, Brief brief,
                            String pa, List<String> va,
                            String pb, List<String> vb,
                            Map<String, PairVerdict> out) {

        char lastLetter = (char) ('A' + vb.size() - 1);
        String relationship = brief.relationships.get(Brief.key(pa, pb));

        StringBuilder u = new StringBuilder();
        if (briefGlosses && !brief.glosses.isEmpty()) {
            u.append("What each option implies:\n").append(brief.glosses).append("\n\n");
        }
        u.append("ONE scenario is being described. Here are two of its facets.\n\n");
        u.append(pa).append(" — pick one:\n");
        for (int a = 0; a < va.size(); a++) {
            u.append("  ").append(a + 1).append(". ").append(va.get(a)).append("\n");
        }
        u.append("\n").append(pb).append(" — pick one:\n");
        for (int b = 0; b < vb.size(); b++) {
            u.append("  ").append((char) ('A' + b)).append(". ").append(vb.get(b)).append("\n");
        }
        u.append("\n");
        if (relationship != null && !relationship.isBlank()) {
            // The specific question this pair of dimensions turns on, as the model itself named it.
            u.append("The question for these two dimensions: ").append(relationship).append("\n");
        }
        u.append("For each of the ").append(va.size() * vb.size())
         .append(" combinations, could one real scenario have BOTH facets together?\n");
        u.append(rule).append("\n\n");

        // These two lines exist because of an observed failure, not as generic prompt padding.
        // On unseen matrices the model excluded entire values against every option of another
        // parameter -- ransomware against every attacker type, rail disruption against every
        // trigger. Those rows are self-evidently wrong and they empty the solution space.
        u.append("Judge every column separately. Most combinations in a real matrix ARE possible, "
               + "so a row that is all N, or a column that is all N, is almost always a mistake — "
               + "re-check it before writing it.\n");
        u.append("Only mark N where you can name the specific reason those two facets cannot go "
               + "together.\n\n");

        u.append("Answer with exactly ").append(va.size()).append(" lines, one per ")
         .append(pa).append(" option in order.\n");
        u.append("Each line: the option number, a colon, then ").append(vb.size())
         .append(" letters (Y or N) for options A to ").append(lastLetter).append(".\n");
        u.append("Example: 1:").append("Y".repeat(vb.size())).append("\n");
        u.append("No other text.");

        StringBuilder g = new StringBuilder("root ::= ");
        for (int a = 0; a < va.size(); a++) {
            g.append("\"").append(a + 1).append(":\" ");
            for (int b = 0; b < vb.size(); b++) g.append("v ");
            g.append("\"\\n\" ");
        }
        g.append("\nv ::= [YN]");

        InferenceParameters ip = new InferenceParameters(buildPrompt(system, u.toString()))
                .setGrammar(g.toString())
                .setTemperature(0.0f)
                .setSeed(42)
                // Every pair shares the same system rubric and domain brief, so caching the prefix
                // avoids re-prefilling all of it once per parameter pair.
                .setCachePrompt(true)
                .setNPredict(va.size() * (vb.size() + 4) + 16);

        // Streamed rather than complete(), because per-token probabilities are only exposed on the
        // streamed output, and they are what turn a bare verdict into a confidence.
        StringBuilder text = new StringBuilder();
        List<Double> pInconsistent = new ArrayList<>();
        if (confidenceScores) ip = ip.setNProbs(2);
        for (LlamaOutput o : model.generate(ip)) {
            String piece = o.text;
            text.append(piece);
            String trimmed = piece.trim();
            if (trimmed.equals("Y") || trimmed.equals("N")) {
                pInconsistent.add(probabilityOfN(o, trimmed));
            } else {
                // A token covering more than one verdict character cannot be attributed per cell,
                // so those cells fall back to the plain verdict at full confidence.
                for (int k = 0; k < trimmed.length(); k++) {
                    char ch = trimmed.charAt(k);
                    if (ch == 'Y') pInconsistent.add(0.0);
                    else if (ch == 'N') pInconsistent.add(1.0);
                }
            }
        }

        int cell = 0;
        for (String line : text.toString().split("\\R")) {
            line = line.trim();
            int c = line.indexOf(':');
            if (c <= 0) continue;
            int row;
            try {
                row = Integer.parseInt(line.substring(0, c).trim()) - 1;
            } catch (NumberFormatException e) {
                continue;
            }
            if (row < 0 || row >= va.size()) continue;

            String verdicts = line.substring(c + 1).trim();
            for (int b = 0; b < vb.size() && b < verdicts.length(); b++) {
                char ch = verdicts.charAt(b);
                if (ch != 'Y' && ch != 'N') continue;
                double p = cell < pInconsistent.size()
                        ? pInconsistent.get(cell) : (ch == 'N' ? 1.0 : 0.0);
                cell++;

                boolean contradiction = p >= confidenceThreshold;
                String detail = String.format(
                        "Local model scored this %.0f%% inconsistent (excluded at %.0f%%), judging %s "
                                + "\"%s\" against %s \"%s\" alongside all %d other %s option(s) in one pass",
                        p * 100, confidenceThreshold * 100,
                        pa, va.get(row), pb, vb.get(b), vb.size() - 1, pb);
                out.put(PairKey.of(pa, va.get(row), pb, vb.get(b)),
                        new PairVerdict(contradiction, p, detail));
            }
        }
    }

    /**
     * P(inconsistent) for one verdict token.
     *
     * <p>The grammar admits only Y and N, so those two probabilities are the whole distribution. If
     * the runtime returns nothing usable, the sampled verdict is taken at full confidence, which
     * reproduces the previous behaviour rather than silently scoring everything 0.
     */
    private static double probabilityOfN(LlamaOutput o, String sampled) {
        Map<String, Float> probs = o.probabilities;
        if (probs == null || probs.isEmpty()) return sampled.equals("N") ? 1.0 : 0.0;

        double n = 0, y = 0;
        for (Map.Entry<String, Float> e : probs.entrySet()) {
            String k = e.getKey().trim();
            if (k.equals("N")) n = e.getValue();
            else if (k.equals("Y")) y = e.getValue();
        }
        double total = n + y;
        if (total <= 0) return sampled.equals("N") ? 1.0 : 0.0;
        return n / total;
    }

    /**
     * Wraps the instruction in the model family's chat template.
     *
     * <p>Gemma 3 has no system role at all -- its template alternates user and model turns only --
     * so the rubric is folded into the opening user turn rather than dropped. Getting the template
     * wrong does not fail loudly: the model just sees its own control tokens as ordinary text and
     * answers noticeably worse, so check it against the model card rather than assuming.
     */
    private String buildPrompt(String system, String user) {
        StringBuilder p = new StringBuilder();
        if ("gemma".equalsIgnoreCase(template)) {
            p.append("<start_of_turn>user\n")
             .append(system).append("\n\n").append(user)
             .append("<end_of_turn>\n<start_of_turn>model\n");
            return p.toString();
        }
        p.append("<|im_start|>system\n").append(system).append("<|im_end|>\n");
        p.append("<|im_start|>user\n").append(user).append("<|im_end|>\n");
        p.append("<|im_start|>assistant\n");
        if (suppressThinking) {
            // For a model that deliberates by default, a closed empty think block is its own
            // documented way of skipping straight to the answer.
            p.append("<think>\n\n</think>\n\n");
        }
        return p.toString();
    }

    /**
     * Removes the characters that carry structural meaning in the prompt and the grammar.
     *
     * <p>A newline inside a value label would split one option across two numbered lines and throw
     * the whole grid out of alignment.
     */
    private static List<String> sanitise(List<String> values) {
        List<String> out = new ArrayList<>();
        if (values == null) return out;
        for (String v : values) {
            if (v == null) continue;
            String clean = v.replaceAll("[\\r\\n|]", " ").trim();
            if (!clean.isEmpty()) out.add(clean);
        }
        return out;
    }

    /** Status for /api/status, so the interface can say what is actually deciding the grid. */
    public Map<String, Object> status() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("enabled", enabled);
        m.put("loaded", isLoaded());
        m.put("model", Paths.get(modelPath).toAbsolutePath().toString());
        m.put("ctxSize", ctxSize);
        m.put("template", template);
        m.put("brief", useBrief);
        m.put("briefGlosses", briefGlosses);
        m.put("confidenceThreshold", confidenceThreshold);
        m.put("criterion", criterion());
        return m;
    }
}
