import * as ts from "typescript";
import * as fs from "fs";
import * as path from "path";

/** Escapes a value so it can be used as a DOT node identifier. */
function escDot(value: string): string {
  return `"${value.replace(/[^A-Za-z0-9_]/g, "_")}"`;
}

/** All supported node kinds. */
export type NodeKind =
  | "function"
  | "class"
  | "method"
  | "arrow"
  | "functionExpression"
  | "objectMethod";

/** A node in the graph with exhaustive metadata. */
export interface GraphNode {
  /** Unique graph key: `{file}:{symbol}`. For external libs, file is 'external'. */
  key: string;
  /** Constructed display name (e.g. `MyClass.myMethod`). */
  name: string;
  /** Actual symbol name (e.g. `myMethod`). */
  actualName: string;
  /** File path relative to project root. */
  file: string;
  /** Inclusive start character index (includes leading comments). */
  start: number;
  /** Inclusive end character index. */
  end: number;
  /** Categorisation of the node. */
  kind: NodeKind;
}

/** A directed edge between two nodes. */
export interface GraphEdge {
  from: string;
  to: string;
  kind: string;
}

/** Main parser capable of emitting DOT or JSON. */
export class CallGraphParser {
  private readonly nodes = new Map<string, GraphNode>();
  private readonly edges: GraphEdge[] = [];
  private root = "";
  private checker!: ts.TypeChecker;
  private currentFile = "";

  /** Parse an entire directory tree of `.ts`/`.tsx` files. */
  parse(directory: string): void {
    this.root = path.resolve(directory);
    const inputFiles = this.collectFiles(this.root);
    const program = ts.createProgram(inputFiles, {
      target: ts.ScriptTarget.ES2020,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.React,
      skipLibCheck: true,
      moduleResolution: ts.ModuleResolutionKind.NodeJs,
    });
    this.checker = program.getTypeChecker();

    for (const abs of inputFiles) {
      const sf = program.getSourceFile(abs);
      if (!sf) continue;
      this.currentFile = path.relative(this.root, abs);
      this.walk(sf);
    }

    // Post-processing step to ensure graph integrity.
    this.ensureEdgeNodesExist();
  }

  /** Return graph in DOT format. */
  toDot(): string {
    const lines = [
      "digraph G {",
      "rankdir=TB;",
      'node [shape=box,style="rounded,filled"];',
    ];
    for (const n of this.nodes.values()) {
      const color =
        n.file === "external" || n.start === 0 ? "lightgray" : "lightblue";
      lines.push(
        `  ${escDot(n.key)} [label="${n.name}\\n(${
          n.file
        })",fillcolor=${color}];`
      );
    }
    for (const e of this.edges)
      lines.push(`  ${escDot(e.from)} -> ${escDot(e.to)};`);
    lines.push("}");
    return lines.join("\n");
  }

  /** Return graph in JSON format (pretty-printed). */
  toJson(indent = 2): string {
    return JSON.stringify(
      { nodes: Array.from(this.nodes.values()), edges: this.edges },
      null,
      indent
    );
  }

  // ────────────────────────── private helpers ──────────────────────────

  /** Depth-first AST walk registering nodes and edges. */
  private walk(node: ts.Node): void {
    this.register(node);

    if (ts.isCallExpression(node)) this.addEdge(node, "call");
    if (ts.isNewExpression(node)) this.addEdge(node, "new");

    if (ts.isJsxElement(node) || ts.isJsxSelfClosingElement(node))
      this.addJsxEdge(node, "JSX");

    if (ts.isJsxAttribute(node)) this.addJsxAttrEdge(node);

    node.forEachChild((c) => this.walk(c));
  }

  /** Register a function/class-like node if it's a declaration. */
  private register(node: ts.Node): void {
    const isDeclaration =
      ts.isFunctionDeclaration(node) ||
      ts.isClassDeclaration(node) ||
      ts.isMethodDeclaration(node) ||
      (ts.isVariableDeclaration(node) &&
        node.initializer &&
        (ts.isArrowFunction(node.initializer) ||
          ts.isFunctionExpression(node.initializer))) ||
      (ts.isPropertyAssignment(node) &&
        node.initializer &&
        (ts.isArrowFunction(node.initializer) ||
          ts.isFunctionExpression(node.initializer)));

    if (!isDeclaration) return;

    const nameNode = ts.getNameOfDeclaration(node as ts.Declaration);
    if (!nameNode) return;

    const symbol = this.checker.getSymbolAtLocation(nameNode);
    if (symbol) {
      this.ensureNodeFromSymbol(symbol);
    }
  }

  /** Record an edge by resolving callee expression. */
  private addEdge(
    node: ts.CallExpression | ts.NewExpression,
    kind: string
  ): void {
    const fromFn = this.enclosingSymbol(node);
    if (!fromFn) return;

    const toSymbol = this.resolveSymbolForExpression(node.expression);
    if (!toSymbol) return;

    const toKey = this.ensureNodeFromSymbol(toSymbol);
    if (!toKey) return;

    const fromKey = `${this.currentFile}:${fromFn}`;
    if (fromKey === toKey) return;

    this.edges.push({ from: fromKey, to: toKey, kind });
  }

  /** Record edges for JSX component usage. */
  private addJsxEdge(
    jsx: ts.JsxElement | ts.JsxSelfClosingElement,
    kind: string
  ): void {
    const tag = ts.isJsxElement(jsx) ? jsx.openingElement.tagName : jsx.tagName;
    if (!ts.isIdentifier(tag) || !/^[A-Z]/.test(tag.text)) return;

    const fromFn = this.enclosingSymbol(jsx);
    if (!fromFn) return;

    const toSymbol = this.resolveSymbolForExpression(tag);
    if (!toSymbol) return;

    const toKey = this.ensureNodeFromSymbol(toSymbol);
    if (toKey) {
      const fromKey = `${this.currentFile}:${fromFn}`;
      this.edges.push({ from: fromKey, to: toKey, kind });
    }
  }

  /** Record attribute handler edges (`onClick={handler}`-style). */
  private addJsxAttrEdge(attr: ts.JsxAttribute): void {
    if (
      !attr.initializer ||
      !ts.isJsxExpression(attr.initializer) ||
      !attr.initializer.expression
    ) {
      return;
    }

    const fromFn = this.enclosingSymbol(attr);
    if (!fromFn) return;

    const toSymbol = this.resolveSymbolForExpression(
      attr.initializer.expression
    );
    if (!toSymbol) return;

    const toKey = this.ensureNodeFromSymbol(toSymbol);
    if (toKey) {
      const fromKey = `${this.currentFile}:${fromFn}`;
      this.edges.push({
        from: fromKey,
        to: toKey,
        kind: "attr",
      });
    }
  }

  /**
   * The core method. Given a symbol, it ensures a node for it exists in the
   * graph and returns its unique key. If the node doesn't exist, it's
   * created on the fly. This prevents orphan edges.
   */
  private ensureNodeFromSymbol(symbol: ts.Symbol): string | null {
    const decl = symbol.declarations?.[0];

    if (!decl) {
      const key = `external:${symbol.getName()}`;
      if (!this.nodes.has(key)) {
        this.nodes.set(key, {
          key,
          name: symbol.getName(),
          actualName: symbol.getName(),
          file: "external",
          start: 0,
          end: 0,
          kind: "function",
        });
      }
      return key;
    }

    const displayName = this.getDisplayNameForDeclaration(decl);
    const file = path.relative(this.root, decl.getSourceFile().fileName);
    const key = `${file}:${displayName}`;

    if (!this.nodes.has(key)) {
      this.nodes.set(key, {
        key,
        name: displayName,
        actualName: symbol.getName(),
        file: file,
        start: decl.getFullStart(),
        end: decl.end,
        kind: this.getKindForDeclaration(decl),
      });
    }
    return key;
  }

  /**
   * Given an expression, find its original symbol. This hybrid
   * approach correctly handles `this.method()`, `object.method()`, and
   * direct function calls.
   */
  private resolveSymbolForExpression(
    expr: ts.Expression
  ): ts.Symbol | undefined {
    let symbol: ts.Symbol | undefined;

    if (ts.isPropertyAccessExpression(expr)) {
      let objectType: ts.Type | undefined;

      if (expr.expression.kind === ts.SyntaxKind.ThisKeyword) {
        objectType = this.checker.getTypeAtLocation(expr.expression);
      } else {
        const objectSymbol = this.checker.getSymbolAtLocation(expr.expression);
        if (objectSymbol) {
          objectType = this.checker.getTypeOfSymbolAtLocation(
            objectSymbol,
            expr.expression
          );
        }
      }

      if (objectType) {
        symbol = this.checker.getPropertyOfType(objectType, expr.name.text);
      }
    } else {
      symbol = this.checker.getSymbolAtLocation(expr);
    }

    if (symbol && symbol.flags & ts.SymbolFlags.Alias) {
      symbol = this.checker.getAliasedSymbol(symbol);
    }
    return symbol;
  }

  /**
   * For a given declaration node, determine its display name.
   */
  private getDisplayNameForDeclaration(node: ts.Node): string {
    if (ts.isMethodDeclaration(node) && ts.isIdentifier(node.name)) {
      const cls = this.enclosingClass(node) ?? "AnonymousClass";
      return `${cls}.${node.name.text}`;
    }
    if (
      ts.isPropertyAssignment(node) &&
      ts.isIdentifier(node.name) &&
      (ts.isArrowFunction(node.initializer) ||
        ts.isFunctionExpression(node.initializer))
    ) {
      let obj = "__object";
      if (
        ts.isObjectLiteralExpression(node.parent) &&
        ts.isVariableDeclaration(node.parent.parent) &&
        ts.isIdentifier(node.parent.parent.name)
      ) {
        obj = node.parent.parent.name.text;
      }
      return `${obj}.${node.name.text}`;
    }
    const nameNode = ts.getNameOfDeclaration(node as ts.Declaration);
    if (nameNode) {
      return nameNode.getText();
    }
    return "anonymous";
  }

  /** For a given declaration, determine its NodeKind. */
  private getKindForDeclaration(node: ts.Node): NodeKind {
    if (ts.isFunctionDeclaration(node)) return "function";
    if (ts.isClassDeclaration(node)) return "class";
    if (ts.isMethodDeclaration(node)) return "method";
    if (ts.isVariableDeclaration(node) && node.initializer) {
      if (ts.isArrowFunction(node.initializer)) return "arrow";
      if (ts.isFunctionExpression(node.initializer))
        return "functionExpression";
    }
    if (
      ts.isPropertyAssignment(node) &&
      node.initializer &&
      (ts.isArrowFunction(node.initializer) ||
        ts.isFunctionExpression(node.initializer))
    ) {
      return "objectMethod";
    }
    return "function";
  }

  /**
   * Get the closest containing function/method name for the 'from' part of an edge.
   */
  private enclosingSymbol(node: ts.Node): string | null {
    for (let p = node.parent; p; p = p.parent) {
      if (ts.isFunctionDeclaration(p) && p.name) {
        return this.getDisplayNameForDeclaration(p);
      }
      if (ts.isMethodDeclaration(p) && ts.isIdentifier(p.name)) {
        return this.getDisplayNameForDeclaration(p);
      }
      if (
        (ts.isVariableDeclaration(p) || ts.isPropertyAssignment(p)) &&
        p.initializer &&
        (ts.isArrowFunction(p.initializer) ||
          ts.isFunctionExpression(p.initializer))
      ) {
        return this.getDisplayNameForDeclaration(p);
      }
    }
    return null;
  }

  /** Get name of enclosing class if any. */
  private enclosingClass(node: ts.Node): string | null {
    for (let p = node.parent; p; p = p.parent)
      if (ts.isClassDeclaration(p) && p.name) return p.name.text;
    return null;
  }

  /** [NEW] Iterates all edges and ensures their from/to nodes exist, creating them if not. */
  private ensureEdgeNodesExist(): void {
    for (const edge of this.edges) {
      this.ensureNodeExists(edge.from);
      this.ensureNodeExists(edge.to);
    }
  }

  /** [NEW] Creates a fallback node for a given key if it doesn't already exist. */
  private ensureNodeExists(key: string): void {
    if (this.nodes.has(key)) {
      return; // Already exists
    }

    // If a node is created here, it means the main parsing logic missed it.
    // We create a fallback node with basic information parsed from the key.
    console.warn(`Creating fallback node for missing key: ${key}`);

    const lastColonIndex = key.lastIndexOf(":");
    const file =
      lastColonIndex > -1 ? key.substring(0, lastColonIndex) : "unknown";
    const name = lastColonIndex > -1 ? key.substring(lastColonIndex + 1) : key;

    const fallbackNode: GraphNode = {
      key,
      name,
      actualName: name,
      file,
      start: 0,
      end: 0,
      kind: "function", // A sensible default
    };

    this.nodes.set(key, fallbackNode);
  }

  /** Recursively collect all TypeScript files (excluding `node_modules`). */
  private collectFiles(dir: string): string[] {
    const out: string[] = [];
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const pth = path.join(dir, e.name);
      if (e.isDirectory() && e.name !== "node_modules")
        out.push(...this.collectFiles(pth));
      else if (e.isFile() && /\.(ts|tsx)$/.test(e.name)) out.push(pth);
    }
    return out;
  }
}
