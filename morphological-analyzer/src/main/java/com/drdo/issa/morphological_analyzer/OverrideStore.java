package com.drdo.issa.morphological_analyzer;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.databind.SerializationFeature;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The analyst's standing rulings on value pairs, persisted to a plain JSON file beside the jar.
 *
 * <p>This is the component that actually encodes domain knowledge. A model reading the strings
 * "Individual" and "Regime Change" has no way to know that regime change requires state-level
 * coercive capacity, because that fact is not in either string. An override records it once, by
 * name, and it holds from then on.
 *
 * <p>Three properties were deliberate:
 *
 * <ul>
 *   <li><b>It outranks everything.</b> An override is applied after the model and after the
 *       calibration head, so a recorded judgement is never silently overturned by either.
 *   <li><b>It survives a re-run.</b> Rulings live on disk, not in browser memory, so re-running the
 *       analysis re-applies them rather than discarding them. This was the single largest
 *       usability defect in the previous build.
 *   <li><b>It is readable.</b> The file is indented JSON with parameter names, value names, verdict
 *       and timestamp spelled out, so a reviewer asking why a combination was eliminated can be
 *       shown the line that eliminated it. "The model said so" is not an auditable answer.
 * </ul>
 *
 * <p>Entries are also the labelled training data for {@link CalibrationService}: every ruling is a
 * human-verified label on a pair, produced as a by-product of ordinary use.
 */
@Service
public class OverrideStore {

    public static final String CONTRADICTION = "CONTRADICTION";
    public static final String COMPATIBLE = "COMPATIBLE";

    /** Where a ruling came from, kept so the audit file distinguishes the two ways they arise. */
    public static final String SOURCE_MANUAL = "MANUAL";
    public static final String SOURCE_GRID = "GRID";

    /**
     * One standing ruling. Public fields, serialised directly by Jackson.
     *
     * <p>Named {@code OverrideEntry} rather than {@code Override} on purpose: a nested type called
     * {@code Override} would shadow {@link java.lang.Override} throughout this file.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class OverrideEntry {
        public String parameterA;
        public String valueA;
        public String parameterB;
        public String valueB;
        /** {@link #CONTRADICTION} or {@link #COMPATIBLE}. */
        public String verdict;
        /** Free text from the analyst explaining the ruling. Optional but strongly encouraged. */
        public String note = "";
        /** {@link #SOURCE_MANUAL} for a single toggle, {@link #SOURCE_GRID} for a verified grid. */
        public String source = SOURCE_MANUAL;
        /** ISO-8601 instant, for the audit trail. */
        public String recordedAt;

        public OverrideEntry() {
        }

        public OverrideEntry(String parameterA, String valueA, String parameterB, String valueB,
                             String verdict, String note, String source) {
            this.parameterA = parameterA;
            this.valueA = valueA;
            this.parameterB = parameterB;
            this.valueB = valueB;
            this.verdict = verdict;
            this.note = note == null ? "" : note;
            this.source = source == null ? SOURCE_MANUAL : source;
            this.recordedAt = Instant.now().toString();
        }

        @JsonIgnore
        public String getKey() {
            return PairKey.of(parameterA, valueA, parameterB, valueB);
        }

        @JsonIgnore
        public boolean isContradiction() {
            return CONTRADICTION.equalsIgnoreCase(verdict);
        }

        @JsonIgnore
        public boolean isValid() {
            return parameterA != null && valueA != null && parameterB != null && valueB != null
                    && verdict != null
                    && (CONTRADICTION.equalsIgnoreCase(verdict) || COMPATIBLE.equalsIgnoreCase(verdict));
        }
    }

    private final Map<String, OverrideEntry> entries = new ConcurrentHashMap<>();
    private final ObjectMapper mapper = JsonMapper.builder().enable(SerializationFeature.INDENT_OUTPUT).build();

    @Value("${gma.overrides.file:data/overrides.json}")
    private String overridesFile;

    @Value("${gma.overrides.enabled:true}")
    private boolean enabled;

    @PostConstruct
    public void init() {
        load();
    }

    public boolean isEnabled() {
        return enabled;
    }

    public Path getPath() {
        return Paths.get(overridesFile).toAbsolutePath();
    }

    // ------------------------------------------------------------------ queries

    /**
     * The standing ruling for a pair, or null if none has been recorded.
     *
     * <p>Order-independent: the pair may be passed either way round.
     */
    public OverrideEntry lookup(String parameterA, String valueA, String parameterB, String valueB) {
        if (!enabled) return null;
        return entries.get(PairKey.of(parameterA, valueA, parameterB, valueB));
    }

    /** Every ruling, newest first. */
    public List<OverrideEntry> all() {
        List<OverrideEntry> out = new ArrayList<>(entries.values());
        out.sort(Comparator.comparing((OverrideEntry e) -> e.recordedAt == null ? "" : e.recordedAt).reversed());
        return out;
    }

    public int size() {
        return entries.size();
    }

    public int countByVerdict(String verdict) {
        int n = 0;
        for (OverrideEntry e : entries.values()) {
            if (verdict.equalsIgnoreCase(e.verdict)) n++;
        }
        return n;
    }

    // ------------------------------------------------------------------ mutations

    /** Records a ruling, replacing any existing one for the same pair. Persists immediately. */
    public synchronized OverrideEntry put(OverrideEntry entry) {
        if (entry == null || !entry.isValid()) {
            throw new IllegalArgumentException(
                    "An override needs parameterA, valueA, parameterB, valueB and a verdict of "
                            + CONTRADICTION + " or " + COMPATIBLE);
        }
        entry.verdict = entry.verdict.toUpperCase();
        if (entry.recordedAt == null) entry.recordedAt = Instant.now().toString();
        if (entry.note == null) entry.note = "";
        if (entry.source == null) entry.source = SOURCE_MANUAL;
        entries.put(entry.getKey(), entry);
        save();
        return entry;
    }

    /**
     * Records many rulings in one pass, persisting once at the end.
     *
     * <p>Used by "save the whole grid as verified", which is what produces a balanced training set:
     * single toggles only ever record pairs the model got wrong, whereas a verified grid records the
     * ones it got right as well.
     *
     * @return how many entries were written
     */
    public synchronized int putAll(List<OverrideEntry> batch) {
        if (batch == null || batch.isEmpty()) return 0;
        int n = 0;
        for (OverrideEntry entry : batch) {
            if (entry == null || !entry.isValid()) continue;
            entry.verdict = entry.verdict.toUpperCase();
            if (entry.recordedAt == null) entry.recordedAt = Instant.now().toString();
            if (entry.note == null) entry.note = "";
            if (entry.source == null) entry.source = SOURCE_GRID;
            // A cell the analyst corrected by hand keeps that provenance when the whole grid is
            // saved over it. Restamping it as GRID would erase the record of who decided it.
            OverrideEntry existing = entries.get(entry.getKey());
            if (existing != null && SOURCE_MANUAL.equalsIgnoreCase(existing.source)) {
                entry.source = SOURCE_MANUAL;
            }
            entries.put(entry.getKey(), entry);
            n++;
        }
        save();
        return n;
    }

    /** Removes the ruling for a pair. Returns true if one was there. */
    public synchronized boolean remove(String parameterA, String valueA, String parameterB, String valueB) {
        boolean removed = entries.remove(PairKey.of(parameterA, valueA, parameterB, valueB)) != null;
        if (removed) save();
        return removed;
    }

    /** Removes every ruling. The file is rewritten as an empty array, not deleted. */
    public synchronized int clear() {
        int n = entries.size();
        entries.clear();
        save();
        return n;
    }

    // ------------------------------------------------------------------ persistence

    private synchronized void load() {
        Path path = getPath();
        if (!Files.exists(path)) {
            System.out.println("ℹ No override file yet at " + path + " (it is created on the first ruling)");
            return;
        }
        try {
            List<OverrideEntry> loaded =
                    mapper.readValue(Files.readAllBytes(path), new TypeReference<List<OverrideEntry>>() {
                    });
            entries.clear();
            for (OverrideEntry e : loaded) {
                if (e != null && e.isValid()) entries.put(e.getKey(), e);
            }
            System.out.println("✅ Loaded " + entries.size() + " analyst override(s) from " + path);
        } catch (Exception e) {
            // A corrupt or unreadable file must not stop the application: the model still works
            // without any overrides, and refusing to start would be worse than starting degraded.
            System.err.println("❌ Could not read overrides from " + path + ": " + e.getMessage());
        }
    }

    /**
     * Writes to a temporary file and moves it into place, so an interrupted write cannot leave a
     * half-written file where the rulings used to be.
     */
    private synchronized void save() {
        Path path = getPath();
        try {
            Path parent = path.getParent();
            if (parent != null) Files.createDirectories(parent);

            List<OverrideEntry> out = new ArrayList<>(entries.values());
            out.sort(Comparator.comparing(OverrideEntry::getKey));

            Path tmp = path.resolveSibling(path.getFileName() + ".tmp");
            Files.write(tmp, mapper.writeValueAsBytes(out));
            try {
                Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicUnsupported) {
                Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (Exception e) {
            System.err.println("❌ Could not save overrides to " + path + ": " + e.getMessage());
        }
    }

    /** Re-reads the file from disk, discarding anything held in memory. */
    public synchronized void reload() {
        entries.clear();
        load();
    }

    /** Rulings in insertion-stable order, for training. */
    public List<OverrideEntry> trainingSet() {
        List<OverrideEntry> out = new ArrayList<>(entries.values());
        out.sort(Comparator.comparing(OverrideEntry::getKey));
        return Collections.unmodifiableList(out);
    }
}
