package com.drdo.issa.morphological_analyzer;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Cross-consistency assessment by the fine-tuned model, over local HTTP.
 *
 * <p><b>Why this is a service call and not another GGUF.</b> {@link LlmCcaService} runs a GGUF
 * through llama.cpp. What the training work produced is a LoRA adapter on Qwen2.5-3B in
 * HuggingFace format, and this machine has no GGUF writer — the {@code gguf} package is absent,
 * which is why the 7B had to be dequantised by hand in the first place. Converting would mean
 * merging the adapter, then writing a GGUF encoder and a Q4_K quantiser from scratch. Calling the
 * model where it already runs gives the same answer, and keeps the deployed path byte-identical to
 * the evaluated one: the prompt lives in one place, on the Python side, imported from the module
 * that built the training data.
 *
 * <p><b>Why the payload carries characterisations.</b> This model cannot judge from value names.
 * Each value must arrive with what it REQUIRES and what it PROVIDES. That is not an interface
 * awkwardness to route around — the format that asked a model to invent its own premises from bare
 * names scored at chance, and moving the premises into the prompt is what made the task learnable.
 * {@link AiController#runFullAnalysis} posts names only, so it cannot feed this service; the
 * dedicated endpoint takes the fuller payload instead.
 *
 * <p>Timeouts are generous on purpose: a sixteen-value box is 160 cells and each is scored in both
 * presentation orders, which is minutes of GPU rather than seconds.
 */
@Service
public class FineTunedCcaService {

    @Value("${gma.cca.url:http://127.0.0.1:8000}")
    private String baseUrl;

    @Value("${gma.cca.enabled:true}")
    private boolean enabled;

    @Value("${gma.cca.timeout-seconds:1800}")
    private int timeoutSeconds;

    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    public boolean isEnabled() {
        return enabled;
    }

    /** Raw JSON from the scoring service, passed straight through to the caller. */
    public String post(String path, String jsonBody) throws Exception {
        if (!enabled) {
            throw new IllegalStateException("gma.cca.enabled=false");
        }
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(Duration.ofSeconds(timeoutSeconds))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (res.statusCode() / 100 != 2) {
            throw new IllegalStateException(
                    "scoring service returned " + res.statusCode() + ": " + res.body());
        }
        return res.body();
    }

    /** Whether the scoring service is up, for /api/status and for a clear error message. */
    public String health() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/health"))
                    .timeout(Duration.ofSeconds(5))
                    .GET()
                    .build();
            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            return res.statusCode() == 200 ? res.body() : "unavailable (" + res.statusCode() + ")";
        } catch (Exception e) {
            return "unreachable at " + baseUrl + " (" + e.getClass().getSimpleName() + ")";
        }
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    /**
     * Assess a whole box with the fine-tuned model, in the same shape
     * {@link LlmCcaService#assessMatrix} returns, so it is a drop-in replacement.
     *
     * <p>The model judges from what each value REQUIRES and PROVIDES, not from its name — the
     * variant that worked from names alone scored at chance, which is why the premises were moved
     * into the prompt. The interface collects names only, so the service generates a
     * characterisation for any value that arrives without one, then scores every cross-parameter
     * pair in both presentation orders.
     *
     * <p>Returns an empty map on any failure, which makes the caller fall back to the other engine
     * rather than losing the grid.
     */
    public Map<String, LlmCcaService.PairVerdict> assessMatrix(
            List<String> parameters, Map<String, List<String>> values) {

        Map<String, LlmCcaService.PairVerdict> out = new HashMap<>();
        if (!enabled) return out;
        long t0 = System.currentTimeMillis();
        try {
            StringBuilder body = new StringBuilder("{\"parameters\":{");
            for (int i = 0; i < parameters.size(); i++) {
                String p = parameters.get(i);
                if (i > 0) body.append(',');
                body.append(q(p)).append(":[");
                List<String> vs = values.getOrDefault(p, List.of());
                for (int j = 0; j < vs.size(); j++) {
                    if (j > 0) body.append(',');
                    body.append(q(vs.get(j)));
                }
                body.append(']');
            }
            body.append("}}");

            String json = post("/assess-box", body.toString());
            for (Map<String, Object> row : pairsOf(json)) {
                String pa = (String) row.get("pa"), a = (String) row.get("a");
                String pb = (String) row.get("pb"), b = (String) row.get("b");
                double p = ((Number) row.get("p_inconsistent")).doubleValue();
                boolean bad = Boolean.TRUE.equals(row.get("contradiction"));
                String detail = bad
                        ? String.format("Fine-tuned model: inconsistent, P=%.2f", p)
                        : String.format("Fine-tuned model: consistent, P(inconsistent)=%.2f", p);
                out.put(PairKey.of(pa, a, pb, b),
                        LlmCcaService.newPairVerdict(bad, p, detail));
            }
            System.out.printf("  fine-tuned CCA: %d pairs in %.1fs%n",
                    out.size(), (System.currentTimeMillis() - t0) / 1000.0);
        } catch (Exception e) {
            System.out.println("  fine-tuned CCA unavailable: " + e.getMessage());
        }
        return out;
    }

    private static String q(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
            }
        }
        return b.append('"').toString();
    }

    /** Minimal extraction of the "pairs" array; avoids adding a JSON dependency. */
    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> pairsOf(String json) throws Exception {
        tools.jackson.databind.ObjectMapper m = new tools.jackson.databind.ObjectMapper();
        Map<String, Object> root = m.readValue(json, Map.class);
        Object p = root.get("pairs");
        return p instanceof List ? (List<Map<String, Object>>) p : List.of();
    }
}
