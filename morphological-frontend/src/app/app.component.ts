import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

/** Bumped if the shape below ever changes, so an old saved session is discarded rather than misread. */
const STATE_KEY = 'gma.state.v1';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  // Pre-loaded parameters from the Johansen Defense Scenario paper
  parameters: any[] = [
    { name: 'ACTOR', valuesStr: 'State, Network, Business Enterprise, Individual', values: ['State', 'Network', 'Business Enterprise', 'Individual'] },
    { name: 'GOAL', valuesStr: 'Regime Change, Political Concessions, Military Exercise and Intelligence Gathering, Economic Gain', values: ['Regime Change', 'Political Concessions', 'Military Exercise and Intelligence Gathering', 'Economic Gain'] },
    { name: 'METHOD', valuesStr: 'Military Control over Entire Territory, Military Control over Parts of Territory, Anti-Access and Area Denial, Symbolic Use of Force, Peace Time Operations, Attack against Infrastructure or Population, Use of Economic Force, Criminality', values: ['Military Control over Entire Territory', 'Military Control over Parts of Territory', 'Anti-Access and Area Denial', 'Symbolic Use of Force', 'Peace Time Operations', 'Attack against Infrastructure or Population', 'Use of Economic Force', 'Criminality'] },
    { name: 'MEANS', valuesStr: 'Large Scale Use of Military Force, Limited Scale Use of Military Force, Large Scale Use of Non-military Force, Limited Scale Use of Non-military Force, Economic Sanctions, Other Means', values: ['Large Scale Use of Military Force', 'Limited Scale Use of Military Force', 'Large Scale Use of Non-military Force', 'Limited Scale Use of Non-military Force', 'Economic Sanctions', 'Other Means'] }
  ];

  analysisData: any = null;
  scenarioDesc: string = '';
  fixedValues: { [key: string]: string } = {};
  bestScenario: { [key: string]: string } | null = null;

  isCalculating: boolean = false;
  /** True once the analyst has changed any cell the model proposed. */
  edited: boolean = false;

  // --- analyst overrides -------------------------------------------------
  overrides: any[] = [];
  overrideCount = 0;
  overrideContradictions = 0;
  overrideCompatible = 0;
  overrideFile = '';
  showOverrides = false;

  // --- calibration head --------------------------------------------------
  calibration: any = null;
  training = false;
  showCalibration = false;

  statusMsg = '';
  restoredSession = false;

  /**
   * Where the API lives.
   *
   * This must never be hardcoded to a port. The failure it causes is quiet and misleading: if
   * the server is started on a port other than the hardcoded one — because that port was already
   * taken — the page still loads from the real server, but every request is sent to the old
   * address. That is a different origin, so the browser blocks it as CORS and the console shows a
   * CORS error, which points at the server when the server was never the problem. Only the
   * target address was wrong.
   *
   * Resolution order, first match wins:
   *   1. ?api=http://host:port/api   — query parameter, for a one-off redirect
   *   2. window.GMA_API_BASE         — set in assets/config.js, for a fixed deployment
   *   3. same origin + /api          — correct whenever the server serves this page, on ANY port
   *
   * Case 3 is the normal one and is port-agnostic by construction: the page and the API share an
   * origin, so moving the server to 8090 moves the API with it and nothing needs changing.
   */
  private api = AppComponent.resolveApiBase();

  private static resolveApiBase(): string {
    const fromQuery = new URLSearchParams(window.location.search).get('api');
    if (fromQuery) {
      return fromQuery.replace(/\/+$/, '');
    }
    const configured = (window as any).GMA_API_BASE;
    if (configured) {
      return String(configured).replace(/\/+$/, '');
    }
    // Angular's dev server (4200) proxies /api to the backend; see proxy.conf.json.
    return '/api';
  }

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.restoreState();
    this.refreshOverrides();
    this.refreshCalibration();
  }

  // ======================================================================
  //  Session persistence
  // ======================================================================

  /**
   * Keeps the working session in browser storage so a refresh, or closing the tab, no longer
   * discards an afternoon of grid work. The solution space is deliberately left out and recomputed
   * on restore: it is the largest part of the payload and is pure combinatorics over the grid, so
   * storing it would risk hitting the storage quota to save no work.
   *
   * Standing rulings are a different matter and live on the server, not here.
   */
  private persist() {
    try {
      const d = this.analysisData;
      const slim = d ? {
        parameters: d.parameters,
        parameterValues: d.parameterValues,
        allValues: d.allValues,
        ccaMatrix: d.ccaMatrix,
        ccaSource: d.ccaSource,
        ccaScore: d.ccaScore,
        ccaDetail: d.ccaDetail,
        totalProblemSpace: d.totalProblemSpace,
        sourceCounts: d.sourceCounts
      } : null;

      localStorage.setItem(STATE_KEY, JSON.stringify({
        parameters: this.parameters,
        analysisData: slim,
        scenarioDesc: this.scenarioDesc,
        fixedValues: this.fixedValues,
        bestScenario: this.bestScenario,
        edited: this.edited,
        savedAt: new Date().toISOString()
      }));
    } catch (e) {
      // A private window, a cleared quota or a browser refusing site data all land here. Losing
      // the convenience of a restored session is not worth interrupting the analyst over.
      console.warn('Could not save session state', e);
    }
  }

  private restoreState() {
    try {
      const raw = localStorage.getItem(STATE_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s.parameters) this.parameters = s.parameters;
      if (s.scenarioDesc) this.scenarioDesc = s.scenarioDesc;
      if (s.fixedValues) this.fixedValues = s.fixedValues;
      if (s.bestScenario) this.bestScenario = s.bestScenario;
      this.edited = !!s.edited;
      if (s.analysisData) {
        this.analysisData = s.analysisData;
        this.recomputeSolutionSpace();
        this.restoredSession = true;
        this.statusMsg = 'Restored the session saved at ' + (s.savedAt || 'an earlier time') + '.';
      }
    } catch (e) {
      console.warn('Could not restore session state', e);
    }
  }

  clearSession() {
    if (!confirm('Discard the grid and results held in this browser?\n\nStanding rulings are stored on the server and are NOT affected.')) return;
    try { localStorage.removeItem(STATE_KEY); } catch (e) { /* nothing useful to do */ }
    this.analysisData = null;
    this.bestScenario = null;
    this.edited = false;
    this.restoredSession = false;
    this.statusMsg = 'Session cleared. Standing rulings are untouched.';
  }

  // ======================================================================
  //  Matrix definition
  // ======================================================================

  updateValues(index: number) {
    this.parameters[index].values = this.parameters[index].valuesStr.split(',').map((v: string) => v.trim());
    this.persist();
  }

  addParameter() {
    this.parameters.push({ name: 'NEW PARAMETER', valuesStr: '', values: [] });
    this.persist();
  }

  getParamMap() {
    let map: any = {};
    this.parameters.forEach(p => {
      if (p.name && p.values.length > 0) {
        map[p.name] = p.values;
      }
    });
    return map;
  }

  // Executes Full Morphological Analysis (Box + Consistency Matrix + Solution Space)
  runFullAnalysis() {
    this.isCalculating = true;
    this.statusMsg = '';
    this.http.post<any>(this.api + '/matrix/full-analysis', this.getParamMap())
      .subscribe({
        next: (res) => {
          this.analysisData = res;
          this.edited = false;
          this.restoredSession = false;
          this.isCalculating = false;
          const c = res.sourceCounts || {};
          const tot = (c.OVERRIDE || 0) + (c.LLM || 0) + (c.CALIBRATED || 0) + (c.MODEL || 0);
          const bits = [];
          if (c.OVERRIDE) bits.push(c.OVERRIDE + ' from your standing rulings');
          if (c.LLM) bits.push(c.LLM + ' assessed by the local model');
          if (c.CALIBRATED) bits.push(c.CALIBRATED + ' from the calibration head');
          if (c.MODEL) bits.push(c.MODEL + ' from the NLI fallback');
          this.statusMsg = 'Assessed ' + tot + ' pairs: ' + (bits.length ? bits.join(', ') : 'none') + '.';
          this.persist();
        },
        error: (err) => {
          console.error(err);
          this.isCalculating = false;
          this.statusMsg = 'Analysis failed. Is the backend running on port 8080?';
        }
      });
  }

  // Calculates most likely scenario strictly from surviving solution space
  matchScenario() {
    if (!this.analysisData || !this.analysisData.solutionSpace) return;

    this.isCalculating = true;
    const payload = {
      scenario: this.scenarioDesc,
      fixedValues: this.fixedValues,
      solutionSpace: this.analysisData.solutionSpace
    };

    this.http.post<any>(this.api + '/matrix/match-scenario', payload)
      .subscribe({
        next: (res) => {
          this.bestScenario = res;
          this.isCalculating = false;
          this.persist();
        },
        error: (err) => {
          console.error(err);
          this.isCalculating = false;
        }
      });
  }

  // ======================================================================
  //  The grid
  // ======================================================================

  /** Mirrors PairKey.java exactly. The two must agree or the browser and server would disagree. */
  private esc(s: string): string {
    return (s === null || s === undefined ? '' : s).replace(/\\/g, '\\\\').replace(/\|/g, '\\|');
  }

  private pairKey(pa: string, va: string, pb: string, vb: string): string {
    const a = this.esc(pa) + '|' + this.esc(va);
    const b = this.esc(pb) + '|' + this.esc(vb);
    return a <= b ? a + '||' + b : b + '||' + a;
  }

  /**
   * Toggles a cell between consistent and inconsistent, records it as a standing ruling, and
   * immediately re-derives the solution space.
   *
   * Cross-consistency assessment is an expert judgement; the model only supplies a first pass, so
   * every cell has to be correctable, the correction has to outlive the session, and the
   * consequences of it have to be visible at once.
   */
  toggleCell(rIndex: number, cIndex: number) {
    const d = this.analysisData;
    const grid = d?.ccaMatrix;
    if (!grid || grid[rIndex][cIndex] === 'BLACK') return;

    const now = grid[rIndex][cIndex] === 'X' ? '' : 'X';
    grid[rIndex][cIndex] = now;
    if (d.ccaSource) d.ccaSource[rIndex][cIndex] = 'OVERRIDE';
    if (d.ccaDetail) d.ccaDetail[rIndex][cIndex] = 'Your ruling, recorded just now';
    this.edited = true;
    this.recomputeSolutionSpace();
    this.persist();

    const rowItem = d.allValues[rIndex];
    const colItem = d.allValues[cIndex];
    this.http.post<any>(this.api + '/overrides', {
      parameterA: rowItem.parameter,
      valueA: rowItem.value,
      parameterB: colItem.parameter,
      valueB: colItem.value,
      verdict: now === 'X' ? 'CONTRADICTION' : 'COMPATIBLE',
      source: 'MANUAL',
      note: ''
    }).subscribe({
      next: () => this.refreshOverrides(),
      error: (err) => {
        console.error(err);
        this.statusMsg = 'The grid was updated here, but the ruling could not be saved to the server.';
      }
    });
  }

  /** Solution space is pure combinatorics, so it is rebuilt in the browser with no round trip. */
  recomputeSolutionSpace() {
    const d = this.analysisData;
    if (!d) return;

    const banned = new Set<string>();
    for (let r = 0; r < d.allValues.length; r++) {
      for (let c = 0; c < d.allValues.length; c++) {
        if (d.ccaMatrix[r][c] === 'X') {
          const a = d.allValues[r];
          const b = d.allValues[c];
          banned.add(this.pairKey(a.parameter, a.value, b.parameter, b.value));
        }
      }
    }

    const params: string[] = d.parameters;
    const solutions: any[] = [];
    const walk = (depth: number, chosen: any) => {
      if (depth === params.length) {
        solutions.push({ ...chosen });
        return;
      }
      const param = params[depth];
      for (const val of d.parameterValues[param]) {
        let ok = true;
        for (const key of Object.keys(chosen)) {
          if (banned.has(this.pairKey(param, val, key, chosen[key]))) { ok = false; break; }
        }
        if (ok) {
          chosen[param] = val;
          walk(depth + 1, chosen);
          delete chosen[param];
        }
      }
    };
    walk(0, {});

    d.solutionSpace = solutions;
    d.solutionSpaceSize = solutions.length;
    d.reductionPercentage = d.totalProblemSpace > 0
      ? (1 - solutions.length / d.totalProblemSpace) * 100
      : 0;
  }

  markedCount(): number {
    const d = this.analysisData;
    if (!d) return 0;
    let n = 0;
    for (let r = 0; r < d.allValues.length; r++)
      for (let c = 0; c < d.allValues.length; c++)
        if (d.ccaMatrix[r][c] === 'X') n++;
    return n;
  }

  clearAllMarks() {
    const d = this.analysisData;
    if (!d) return;
    if (!confirm('Clear every mark in the grid?\n\nThis changes the grid on screen only. Your standing rulings on the server are not deleted - use the Standing rulings panel for that.')) return;
    for (let r = 0; r < d.allValues.length; r++)
      for (let c = 0; c < d.allValues.length; c++)
        if (d.ccaMatrix[r][c] === 'X') d.ccaMatrix[r][c] = '';
    this.edited = true;
    this.recomputeSolutionSpace();
    this.persist();
  }

  /** Colour carries provenance: the analyst's own cells are darker than the machine's guesses. */
  cellColor(r: number, c: number): string {
    const d = this.analysisData;
    const v = d.ccaMatrix[r][c];
    if (v === 'BLACK') return '#1a1a1a';
    const src = d.ccaSource ? d.ccaSource[r][c] : '';
    if (v === 'X') return src === 'OVERRIDE' ? '#f1aeb5' : '#fdeaea';
    return src === 'OVERRIDE' ? '#d1e7dd' : '#ffffff';
  }

  cellTitle(r: number, c: number): string {
    const d = this.analysisData;
    if (d.ccaMatrix[r][c] === 'BLACK') return '';
    const rowItem = d.allValues[r];
    const colItem = d.allValues[c];
    let t = rowItem.parameter + ': ' + rowItem.value + '\n' + colItem.parameter + ': ' + colItem.value;
    if (d.ccaDetail && d.ccaDetail[r][c]) t += '\n\n' + d.ccaDetail[r][c];
    t += '\n\nClick to change.';
    return t;
  }

  // ======================================================================
  //  Standing rulings
  // ======================================================================

  refreshOverrides() {
    this.http.get<any>(this.api + '/overrides').subscribe({
      next: (res) => {
        this.overrides = res.entries || [];
        this.overrideCount = res.count || 0;
        this.overrideContradictions = res.contradictionCount || 0;
        this.overrideCompatible = res.compatibleCount || 0;
        this.overrideFile = res.file || '';
      },
      error: (err) => console.error(err)
    });
  }

  /**
   * Records every assessable cell as a ruling in one go.
   *
   * This is what makes the calibration head trainable. Toggling single cells only ever records the
   * pairs the model got wrong, which is a biased sample; saving a grid the analyst has been through
   * records the pairs it got right as well, so both classes are represented.
   */
  saveGridAsVerified() {
    const d = this.analysisData;
    if (!d) return;
    const batch: any[] = [];
    for (let r = 0; r < d.allValues.length; r++) {
      for (let c = 0; c < d.allValues.length; c++) {
        if (d.ccaMatrix[r][c] === 'BLACK') continue;
        const a = d.allValues[r];
        const b = d.allValues[c];
        batch.push({
          parameterA: a.parameter, valueA: a.value,
          parameterB: b.parameter, valueB: b.value,
          verdict: d.ccaMatrix[r][c] === 'X' ? 'CONTRADICTION' : 'COMPATIBLE',
          source: 'GRID',
          note: ''
        });
      }
    }
    if (!batch.length) return;
    if (!confirm('Record all ' + batch.length + ' assessed pairs in this grid as your verified judgement?\n\nGo through the grid and correct it first - every cell is saved exactly as it currently stands.')) return;

    this.http.post<any>(this.api + '/overrides/bulk', batch).subscribe({
      next: (res) => {
        this.statusMsg = 'Recorded ' + res.written + ' rulings. ' + res.count + ' now held in total.';
        this.refreshOverrides();
      },
      error: (err) => {
        console.error(err);
        this.statusMsg = 'Could not record the grid.';
      }
    });
  }

  deleteOverride(entry: any) {
    this.http.post<any>(this.api + '/overrides/delete', entry).subscribe({
      next: () => {
        this.statusMsg = 'Ruling removed. That pair returns to being decided by the model on the next run.';
        this.refreshOverrides();
      },
      error: (err) => console.error(err)
    });
  }

  clearOverrides() {
    if (!confirm('Delete every standing ruling?\n\nThis erases the accumulated domain knowledge on the server and cannot be undone.')) return;
    this.http.post<any>(this.api + '/overrides/clear', {}).subscribe({
      next: (res) => {
        this.statusMsg = 'Deleted ' + res.removed + ' rulings.';
        this.refreshOverrides();
      },
      error: (err) => console.error(err)
    });
  }

  // ======================================================================
  //  Calibration head
  // ======================================================================

  refreshCalibration() {
    this.http.get<any>(this.api + '/calibration').subscribe({
      next: (res) => this.calibration = res,
      error: (err) => console.error(err)
    });
  }

  trainCalibration() {
    this.training = true;
    this.statusMsg = 'Fitting the calibration head. Each ruling needs two model passes, so this takes a moment.';
    this.http.post<any>(this.api + '/calibration/train', {}).subscribe({
      next: (res) => {
        this.training = false;
        this.calibration = res.status;
        this.statusMsg = res.message;
      },
      error: (err) => {
        console.error(err);
        this.training = false;
        this.statusMsg = 'Training failed.';
      }
    });
  }

  resetCalibration() {
    if (!confirm('Discard the fitted calibration head?\n\nPairs go back to being decided by the raw model threshold. Your standing rulings are not affected.')) return;
    this.http.post<any>(this.api + '/calibration/reset', {}).subscribe({
      next: (res) => {
        this.calibration = res.status;
        this.statusMsg = 'Calibration head discarded.';
      },
      error: (err) => console.error(err)
    });
  }

  pct(x: number): string {
    return x === null || x === undefined || x < 0 ? 'n/a' : (x * 100).toFixed(1) + '%';
  }

  // ======================================================================
  //  Over-pruning rescue
  // ======================================================================

  /**
   * Finds values that cannot appear in ANY configuration.
   *
   * A value is dead when it has been marked inconsistent against *every* value of some other
   * parameter: there is then no way to complete a configuration containing it, so it single-handedly
   * removes a whole slice of the solution space. When the model over-prunes, this is almost always
   * why, and one or two such values are usually the entire cause of an empty result.
   *
   * This matters most in front of an audience. "No configurations survive" is a dead end unless the
   * tool can say which mark caused it and undo that mark in one click.
   */
  deadValues(): any[] {
    const d = this.analysisData;
    if (!d || !d.allValues) return [];

    const idx = new Map<string, number>();
    d.allValues.forEach((v: any, i: number) => idx.set(v.parameter + '\u0000' + v.value, i));

    const out: any[] = [];
    for (const item of d.allValues) {
      const i = idx.get(item.parameter + '\u0000' + item.value);
      if (i === undefined) continue;
      for (const q of d.parameters) {
        if (q === item.parameter) continue;
        const qVals = d.parameterValues[q] || [];
        if (!qVals.length) continue;
        let allBlocked = true;
        for (const qv of qVals) {
          const j = idx.get(q + '\u0000' + qv);
          if (j === undefined) { allBlocked = false; break; }
          const marked = d.ccaMatrix[i][j] === 'X' || d.ccaMatrix[j][i] === 'X';
          if (!marked) { allBlocked = false; break; }
        }
        if (allBlocked) {
          out.push({ parameter: item.parameter, value: item.value, blockedBy: q, count: qVals.length });
          break;
        }
      }
    }
    return out;
  }

  /**
   * Clears every mark between one value and all values of the parameter that is blocking it, so the
   * value becomes reachable again. Records the cleared cells as standing rulings, exactly as
   * clicking each of them by hand would.
   */
  freeValue(dead: any) {
    const d = this.analysisData;
    if (!d) return;

    const idx = new Map<string, number>();
    d.allValues.forEach((v: any, i: number) => idx.set(v.parameter + '\u0000' + v.value, i));
    const i = idx.get(dead.parameter + '\u0000' + dead.value);
    if (i === undefined) return;

    const batch: any[] = [];
    for (const qv of (d.parameterValues[dead.blockedBy] || [])) {
      const j = idx.get(dead.blockedBy + '\u0000' + qv);
      if (j === undefined) continue;
      for (const [r, c] of [[i, j], [j, i]]) {
        if (d.ccaMatrix[r][c] === 'X') {
          d.ccaMatrix[r][c] = '';
          if (d.ccaSource) d.ccaSource[r][c] = 'OVERRIDE';
          if (d.ccaDetail) d.ccaDetail[r][c] = 'Cleared to restore the solution space';
        }
      }
      batch.push({
        parameterA: dead.parameter, valueA: dead.value,
        parameterB: dead.blockedBy, valueB: qv,
        verdict: 'COMPATIBLE', source: 'MANUAL',
        note: 'Cleared because ' + dead.value + ' was excluded against every ' + dead.blockedBy
      });
    }

    this.edited = true;
    this.recomputeSolutionSpace();
    this.persist();

    this.http.post<any>(this.api + '/overrides/bulk', batch).subscribe({
      next: () => {
        this.statusMsg = 'Freed "' + dead.value + '" against ' + dead.blockedBy
          + '. Solution space is now ' + this.analysisData.solutionSpaceSize + '.';
        this.refreshOverrides();
      },
      error: (err) => console.error(err)
    });
  }

  getMaxRows(): number[] {
    let max = 0;
    this.parameters.forEach(p => {
      if (p.values.length > max) max = p.values.length;
    });
    return Array.from({ length: max }, (_, i) => i);
  }
}
