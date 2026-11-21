// Parameter acceptance (params.ts). The solver's stated posture is fail-loud:
// a keyword it cannot honour must stop the run, never be read and quietly
// ignored. Four paths broke that and are pinned here, plus the sweep that
// proves the strictness costs no real case file.
//
// Found by the 2026-09-18 scope audit; see SCOPE_REGISTER.md §3 (E9, E10).
import { assertEquals, assertThrows } from "@std/assert";
import { describeParams, parseParamsTxt, resolveParams } from "../src/params.ts";

const CASES = new URL("../verification/cases", import.meta.url).pathname;

/** A real case file, as the raw key/value map, for mutation per test. */
async function raw(caseName: string): Promise<Record<string, string>> {
  return parseParamsTxt(await Deno.readTextFile(`${CASES}/${caseName}/params.txt`));
}

Deno.test("every verification case still parses", async () => {
  const bad: string[] = [];
  let n = 0;
  for await (const e of Deno.readDir(CASES)) {
    if (!e.isDirectory) continue;
    const f = `${CASES}/${e.name}/params.txt`;
    try { await Deno.stat(f); } catch { continue; }
    n++;
    try { resolveParams(parseParamsTxt(await Deno.readTextFile(f))); }
    catch (err) { bad.push(`${e.name}: ${(err as Error).message}`); }
  }
  assertEquals(bad, [], `cases rejected by the parameter checks:\n${bad.join("\n")}`);
  assertEquals(n > 0, true, "no case files found — check the path");
});

Deno.test("an unrecognised name is refused, not ignored", async () => {
  const p = await raw("sb1d_tide");
  p["gammma"] = "0.55"; // the misspelling that used to do nothing at all
  assertThrows(() => resolveParams(p), Error, "unrecognised parameter");
});

Deno.test("windfile is refused rather than run with no wind", async () => {
  const p = await raw("sb1d_tide");
  p["wind"] = "1";
  p["windfile"] = "wind.txt";
  // it used to leave windv at 0, and the momentum kernels gate wind stress
  // on windv > 0 — so the run proceeded with no wind and said nothing
  assertThrows(() => resolveParams(p), Error, "windfile is not implemented");
});

Deno.test("a constant wind is still accepted", async () => {
  const p = await raw("sb1d_tide");
  p["wind"] = "1";
  p["windv"] = "12";
  p["windth"] = "300";
  const out = resolveParams(p);
  assertEquals(out.windv, 12);
  assertEquals(out.windth, 300);
});

Deno.test("a spatially varying tide is refused, not flattened", async () => {
  const p = await raw("sb1d_tide");
  p["tideloc"] = "2";
  // tideloc 2-4 used to run as ONE uniform level from the first zs column
  assertThrows(() => resolveParams(p), Error, "not implemented");
});

Deno.test("the uniform tide path still works", async () => {
  const out = resolveParams(await raw("sb1d_tide"));
  assertEquals(out.tideloc, 1);
  assertEquals(out.zs0file, "tide.txt");
});

Deno.test("an out-of-scope process on is refused, sws included", async () => {
  for (const k of ["sedtrans", "morphology", "sws"]) {
    const p = await raw("sb1d_tide");
    p[k] = "1";
    // sws sat in the out-of-scope list but was explicitly exempted from the
    // guard that reads it — the list entry was wrong, or the guard was
    assertThrows(() => resolveParams(p), Error, "out of scope", `${k}=1 was accepted`);
  }
});

// OCE-288: both were accepted and never loaded — fwfile fell back to the
// constant fw, bedfricfile left the coefficient at its -999 placeholder
Deno.test("per-cell friction files are refused, not ignored", async () => {
  for (const k of ["fwfile", "bedfricfile"]) {
    const p = await raw("sb1d_tide");
    p[k] = "friction.dep";
    if (k === "bedfricfile") p.bedfriction = "manning";
    assertThrows(() => resolveParams(p), Error, "not implemented", `${k} was accepted`);
  }
});

Deno.test("out-of-scope processes explicitly off are fine", async () => {
  const p = await raw("sb1d_tide");
  p["sedtrans"] = "0";
  p["morphology"] = "0";
  resolveParams(p); // must not throw, and must not report them as unknown
});

Deno.test("OCE-215: the effective parameter set names what the file set and what defaulted", async () => {
  const p = await raw("sb1d_tide");
  const rec = describeParams(p, resolveParams(p));
  const by = new Map(rec.map((r) => [r.name, r]));
  // a keyword the file carries is marked set, with the file's value
  const nx = by.get("nx")!;
  assertEquals(nx.set, true);
  assertEquals(nx.value, parseInt(p["nx"], 10));
  // a physics default the file does not mention is present and marked default
  const gamma = by.get("gamma")!;
  assertEquals(gamma.set, "gamma" in p);
  assertEquals(typeof gamma.value, "number");
  // internal derived fields are not XBeach keywords and are left out
  assertEquals(by.has("rhog8"), false);
  assertEquals(by.has("px"), false);
  // a coded choice reads as the name params.txt takes, not the code
  assertEquals(typeof by.get("break")!.value, "string");
  assertEquals(["roelvink1", "baldock", "roelvink2", "roelvink_daly", "janssen"].includes(by.get("break")!.value as string), true);
  assertEquals(by.get("wavemodel")!.value, "surfbeat");
  // every name is unique
  assertEquals(new Set(rec.map((r) => r.name)).size, rec.length);
});
