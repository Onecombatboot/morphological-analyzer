package com.drdo.issa.morphological_analyzer;

/**
 * Canonical, order-independent identity for a pair of morphological values.
 *
 * <p>Two properties matter here, and the original value-only hash had neither.
 *
 * <p><b>Parameters are part of the identity.</b> Keying on values alone means the same value string
 * appearing under two different parameters shares one record, so a contradiction recorded for one is
 * silently applied to the other. Including the parameter name makes each cell of the grid distinct.
 *
 * <p><b>The separator is escaped.</b> A value containing the separator could otherwise collide with
 * a different pair. Backslash and pipe are escaped inside each half, so the {@code ||} joining the
 * two halves cannot occur anywhere inside them and the key parses back unambiguously.
 *
 * <p>The two halves are sorted before joining, so (A, B) and (B, A) produce the same key. Cross
 * consistency is symmetric, so a ruling recorded in one direction must be found in the other.
 */
public final class PairKey {

    private PairKey() {
    }

    /** Escapes the two characters that carry structural meaning in a key. */
    private static String escape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("|", "\\|");
    }

    /** Half of a key: one parameter and one of its values. Cannot contain the {@code ||} joiner. */
    private static String half(String parameter, String value) {
        return escape(parameter) + "|" + escape(value);
    }

    /**
     * Order-independent key for a (parameter, value) against a (parameter, value).
     *
     * @return a stable string that is identical whichever way round the two sides are passed
     */
    public static String of(String parameterA, String valueA, String parameterB, String valueB) {
        String a = half(parameterA, valueA);
        String b = half(parameterB, valueB);
        return a.compareTo(b) <= 0 ? a + "||" + b : b + "||" + a;
    }

    /**
     * Whether the A side sorts first under the same ordering {@link #of} uses.
     *
     * <p>This exists so the calibration head can put a pair into one fixed orientation before
     * extracting features. Without it the head is trained on features computed in whatever order a
     * ruling happened to be stored, and then applied to features computed in parameter-declaration
     * order — that is, trained on one orientation and used on the other for roughly half of all
     * pairs. The raw model still reads pairs in declaration order, because a single correctly
     * oriented reading measured best for it; the head is different, because it is handed both
     * directions as features either way and so only needs the two ends to be consistent.
     */
    public static boolean aSortsFirst(String parameterA, String valueA, String parameterB, String valueB) {
        return half(parameterA, valueA).compareTo(half(parameterB, valueB)) <= 0;
    }
}
