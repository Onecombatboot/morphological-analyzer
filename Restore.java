import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.zip.*;

/**
 * Rebuilds the delivery from its archive volumes. Java replacement for restore.py.
 *
 * <p>Run it directly, with no compile step:
 *
 * <pre>
 *     java Restore.java
 *     java Restore.java --parts D:\gma --out C:\gma
 * </pre>
 *
 * <p>Java 11 and later run a single source file straight from source, so nothing
 * here needs javac, Maven, or a build directory. That is the whole point: the
 * machine this runs on is expected to have a Java runtime and nothing else. No
 * Python, no pip, no network.
 *
 * <p>It does the same three things restore.py did, in the same order:
 * <ol>
 *   <li>extract every entry from every {@code gma_part_*.zip},</li>
 *   <li>rejoin files that were split across volumes to keep each under 1.5 GB,</li>
 *   <li>verify every rebuilt file against the size and SHA-256 in MANIFEST.json.</li>
 * </ol>
 *
 * <p>A mismatch is reported per file and counted; it does not abort the run,
 * because knowing which files are bad is more useful than stopping at the first.
 */
public class Restore {

    // ---- tiny JSON reader ---------------------------------------------------
    // The manifest is machine-written and its shape is fixed, so a full JSON
    // library would be a dependency for no benefit. This handles exactly the
    // subset the manifest uses: objects, arrays, strings, numbers.
    static final class Json {
        private final String s;
        private int i;

        Json(String s) { this.s = s; }

        static Object parse(String text) {
            Json p = new Json(text);
            p.ws();
            return p.value();
        }

        private void ws() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        }

        private Object value() {
            char c = s.charAt(i);
            switch (c) {
                case '{': return object();
                case '[': return array();
                case '"': return string();
                case 't': i += 4; return Boolean.TRUE;
                case 'f': i += 5; return Boolean.FALSE;
                case 'n': i += 4; return null;
                default:  return number();
            }
        }

        private Map<String, Object> object() {
            Map<String, Object> m = new LinkedHashMap<>();
            i++; ws();
            if (s.charAt(i) == '}') { i++; return m; }
            while (true) {
                ws();
                String k = string();
                ws();
                i++;                       // ':'
                ws();
                m.put(k, value());
                ws();
                if (s.charAt(i) == ',') { i++; continue; }
                i++;                       // '}'
                return m;
            }
        }

        private List<Object> array() {
            List<Object> l = new ArrayList<>();
            i++; ws();
            if (s.charAt(i) == ']') { i++; return l; }
            while (true) {
                ws();
                l.add(value());
                ws();
                if (s.charAt(i) == ',') { i++; continue; }
                i++;                       // ']'
                return l;
            }
        }

        private String string() {
            StringBuilder b = new StringBuilder();
            i++;                           // opening quote
            while (true) {
                char c = s.charAt(i++);
                if (c == '"') return b.toString();
                if (c != '\\') { b.append(c); continue; }
                char e = s.charAt(i++);
                switch (e) {
                    case 'n': b.append('\n'); break;
                    case 't': b.append('\t'); break;
                    case 'r': b.append('\r'); break;
                    case 'b': b.append('\b'); break;
                    case 'f': b.append('\f'); break;
                    case 'u':
                        b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                        break;
                    default:  b.append(e);
                }
            }
        }

        private Object number() {
            int st = i;
            while (i < s.length() && "+-.eE0123456789".indexOf(s.charAt(i)) >= 0) i++;
            String t = s.substring(st, i);
            if (t.contains(".") || t.contains("e") || t.contains("E")) return Double.parseDouble(t);
            return Long.parseLong(t);
        }
    }

    // ---- helpers ------------------------------------------------------------
    static String sha256(Path p) throws Exception {
        MessageDigest d = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(Files.newInputStream(p), 1 << 20)) {
            byte[] buf = new byte[1 << 20];
            int n;
            while ((n = in.read(buf)) > 0) d.update(buf, 0, n);
        }
        StringBuilder sb = new StringBuilder();
        for (byte b : d.digest()) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    static String human(long bytes) {
        if (bytes >= 1L << 30) return String.format("%.2f GB", bytes / (double) (1L << 30));
        if (bytes >= 1L << 20) return String.format("%.1f MB", bytes / (double) (1L << 20));
        return bytes + " B";
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asMap(Object o) { return (Map<String, Object>) o; }

    @SuppressWarnings("unchecked")
    static List<Object> asList(Object o) { return (List<Object>) o; }

    static long asLong(Object o) { return o instanceof Double ? (long) (double) (Double) o : (Long) o; }

    // ---- main ---------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        Path parts = Paths.get(".");
        Path out = Paths.get(".");
        for (int k = 0; k < args.length - 1; k++) {
            if (args[k].equals("--parts")) parts = Paths.get(args[k + 1]);
            if (args[k].equals("--out")) out = Paths.get(args[k + 1]);
        }

        System.out.println();
        System.out.println("  Restoring from " + parts.toAbsolutePath());
        System.out.println("  Writing to     " + out.toAbsolutePath());
        System.out.println();

        List<Path> vols = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(parts, "gma_part_*.zip")) {
            for (Path p : ds) vols.add(p);
        }
        Collections.sort(vols);
        if (vols.isEmpty()) {
            System.out.println("  No gma_part_*.zip found. Point --parts at the folder holding them.");
            System.exit(1);
        }
        System.out.println("  Found " + vols.size() + " volume(s)");

        // manifest: prefer the loose copy, fall back to the one inside volume 1
        String manifestText = null;
        Path loose = parts.resolve("MANIFEST.json");
        if (Files.exists(loose)) {
            manifestText = new String(Files.readAllBytes(loose), "UTF-8");
        } else {
            try (ZipFile z = new ZipFile(vols.get(0).toFile())) {
                ZipEntry e = z.getEntry("MANIFEST.json");
                if (e != null) {
                    manifestText = new String(z.getInputStream(e).readAllBytes(), "UTF-8");
                }
            }
        }
        if (manifestText == null) {
            System.out.println("  MANIFEST.json not found, in this folder or inside the first volume.");
            System.exit(1);
        }
        Map<String, Object> manifest = asMap(Json.parse(manifestText));

        // ---- 1. extract -----------------------------------------------------
        System.out.println();
        System.out.println("  [1/3] Extracting");
        long written = 0;
        int files = 0;
        for (Path v : vols) {
            int n = 0;
            try (ZipFile z = new ZipFile(v.toFile())) {
                Enumeration<? extends ZipEntry> en = z.entries();
                while (en.hasMoreElements()) {
                    ZipEntry e = en.nextElement();
                    String name = e.getName();
                    if (e.isDirectory()) continue;
                    // MANIFEST.json is read separately and is not payload.
                    // Restore.java IS extracted: it is listed in the manifest,
                    // so skipping it would report it missing at verification,
                    // and a restored tree that contains the tool that restored
                    // it is worth having.
                    if (name.equals("MANIFEST.json")) continue;
                    Path target = out.resolve(name).normalize();
                    if (!target.startsWith(out.toAbsolutePath().normalize())
                            && !target.startsWith(out.normalize())) {
                        System.out.println("      refusing path outside the output folder: " + name);
                        continue;
                    }
                    Files.createDirectories(target.getParent());
                    try (InputStream in = z.getInputStream(e);
                         OutputStream os = new BufferedOutputStream(
                                 Files.newOutputStream(target), 1 << 20)) {
                        byte[] buf = new byte[1 << 20];
                        int r;
                        while ((r = in.read(buf)) > 0) { os.write(buf, 0, r); written += r; }
                    }
                    n++; files++;
                }
            }
            System.out.println("      " + v.getFileName() + "  " + n + " entries");
        }
        System.out.println("      " + files + " files, " + human(written));

        // ---- 2. rejoin split files -----------------------------------------
        List<Object> split = manifest.containsKey("split") ? asList(manifest.get("split"))
                                                           : Collections.emptyList();
        System.out.println();
        System.out.println("  [2/3] Rejoining " + split.size() + " split file(s)");
        for (Object so : split) {
            Map<String, Object> sp = asMap(so);
            String path = ((String) sp.get("path")).replace('\\', '/');
            List<Object> pieces = asList(sp.get("pieces"));
            Path target = out.resolve(path);
            Files.createDirectories(target.getParent());
            long total = 0;
            try (OutputStream os = new BufferedOutputStream(
                    Files.newOutputStream(target), 1 << 20)) {
                for (Object po : pieces) {
                    String pname = ((String) asMap(po).get("name")).replace('\\', '/');
                    Path piece = out.resolve(pname);
                    if (!Files.exists(piece)) {
                        System.out.println("      MISSING PIECE " + pname);
                        continue;
                    }
                    try (InputStream in = new BufferedInputStream(
                            Files.newInputStream(piece), 1 << 20)) {
                        byte[] buf = new byte[1 << 20];
                        int r;
                        while ((r = in.read(buf)) > 0) { os.write(buf, 0, r); total += r; }
                    }
                    Files.delete(piece);
                }
            }
            System.out.println("      " + path + "  " + human(total));
        }

        // ---- 3. verify ------------------------------------------------------
        System.out.println();
        System.out.println("  [3/3] Verifying against the manifest");
        int checked = 0, bad = 0;
        List<Map<String, Object>> all = new ArrayList<>();
        for (Object o : asList(manifest.get("files"))) all.add(asMap(o));
        for (Object o : split) all.add(asMap(o));

        for (Map<String, Object> f : all) {
            String path = ((String) f.get("path")).replace('\\', '/');
            Path p = out.resolve(path);
            if (!Files.exists(p)) {
                System.out.println("      MISSING   " + path);
                bad++;
                continue;
            }
            long want = asLong(f.get("size"));
            long got = Files.size(p);
            if (got != want) {
                System.out.printf("      SIZE      %s  expected %d, got %d%n", path, want, got);
                bad++;
                continue;
            }
            Object wantHash = f.get("sha256");
            if (wantHash != null) {
                String h = sha256(p);
                if (!h.equalsIgnoreCase((String) wantHash)) {
                    System.out.println("      SHA256    " + path);
                    bad++;
                    continue;
                }
            }
            checked++;
            if (checked % 50 == 0) System.out.println("      " + checked + " verified ...");
        }

        System.out.println();
        System.out.println("  ============================================================");
        if (bad == 0) {
            System.out.println("  RESTORE COMPLETE -- " + checked + " files, 0 mismatches");
        } else {
            System.out.println("  RESTORE FINISHED WITH " + bad + " PROBLEM(S) -- "
                    + checked + " good");
            System.out.println("  Re-copy the affected volume and run this again; it is safe to repeat.");
        }
        System.out.println("  ============================================================");
        System.out.println();
        System.exit(bad == 0 ? 0 : 1);
    }
}
