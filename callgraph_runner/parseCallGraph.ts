#!/usr/bin/env node
// parseCallGraph.ts – CLI wrapper supporting DOT or JSON and stdout fallback.

import { CallGraphParser } from "./callGraphParser";
import * as fs from "fs";
import * as path from "path";

/** Parse CLI arguments. */
function parseArgs() {
  let format: "dot" | "json" = "dot";
  let dir = "";
  let outfile: string | null = null;

  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "-f" || arg === "--format") {
      const val = argv[++i];
      if (val === "json" || val === "dot") format = val;
      else throw new Error(`Unknown format: ${val}`);
    } else if (!dir) dir = arg;
    else if (!outfile) outfile = arg;
  }

  if (!dir)
    throw new Error(
      "Usage: parseCallGraph <projectDir> [-f json|dot] [outputFile]"
    );

  return { dir, format, outfile };
}

/** Entry point. */
function main() {
  try {
    const { dir, format, outfile } = parseArgs();
    if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory())
      throw new Error(`Directory not found: ${dir}`);

    const parser = new CallGraphParser();
    parser.parse(dir);

    const result = format === "json" ? parser.toJson() : parser.toDot();

    if (outfile) fs.writeFileSync(outfile, result);
    else process.stdout.write(result);
  } catch (e) {
    console.error((e as Error).message);
    process.exit(1);
  }
}

if (require.main === module) main();
