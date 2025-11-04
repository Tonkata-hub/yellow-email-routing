import OpenAI from "openai";
import * as fs from "fs";
import * as readline from "readline";
import "dotenv/config";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const MODEL = "text-embedding-3-small"; // 1536-dim
const THRESHOLD = 0.4; // below => "unclassified"

type Centroids = Record<string, number[]>;

function normalize(v: number[]): number[] {
	const len = Math.hypot(...v) + 1e-12;
	return v.map((x) => x / len);
}

function cosine(a: number[], b: number[]): number {
	// with normalized vectors, cosine == dot product
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

async function embedTexts(texts: string[]): Promise<number[][]> {
	const res = await client.embeddings.create({ model: MODEL, input: texts });
	return res.data.map((d) => normalize(d.embedding as number[]));
}

async function buildCentroids(): Promise<void> {
	const raw = fs.readFileSync("emails.json", "utf-8");
	const emails: { text: string; label: string }[] = JSON.parse(raw);
	if (!emails.length) throw new Error("emails.json is empty");

	const texts = emails.map((e) => e.text);
	const labels = emails.map((e) => e.label);
	const vecs = await embedTexts(texts);

	// group vectors by label
	const groups = new Map<string, number[][]>();
	labels.forEach((lab, i) => {
		const arr = groups.get(lab) ?? [];
		arr.push(vecs[i]);
		groups.set(lab, arr);
	});

	// mean vector per label -> centroid
	const centroids: Centroids = {};
	for (const [lab, vs] of groups) {
		const dim = vs[0].length;
		const mean = new Array(dim).fill(0);
		for (const v of vs) for (let j = 0; j < dim; j++) mean[j] += v[j];
		for (let j = 0; j < dim; j++) mean[j] /= vs.length;
		centroids[lab] = normalize(mean);
	}

	fs.writeFileSync("centroids.json", JSON.stringify(centroids, null, 2));
	console.log("✅ Wrote centroids.json for labels:", Object.keys(centroids));
}

async function classifyText(text: string): Promise<void> {
	if (!fs.existsSync("centroids.json")) {
		console.log("No centroids.json found. Run: npm run build");
		process.exit(1);
	}
	const centroids: Centroids = JSON.parse(fs.readFileSync("centroids.json", "utf-8"));
	const [emb] = await embedTexts([text]);

	const sims = Object.entries(centroids).map(([lab, c]) => [lab, cosine(emb, c)] as const);
	sims.sort((a, b) => b[1] - a[1]);

	const [bestLabel, bestScore] = sims[0];
	const routed = bestScore >= THRESHOLD ? bestLabel : "unclassified";

	console.log(`📩 Routed to: ${routed} (best=${bestLabel}, score=${bestScore.toFixed(3)})`);
	console.log("Similarities:", Object.fromEntries(sims.map(([l, s]) => [l, +s.toFixed(3)])));
}

async function classifyFromStdin(): Promise<void> {
	const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
	console.log("Paste the email text (end with Ctrl+D):");
	let text = "";
	for await (const line of rl) text += line + "\n";
	rl.close();
	await classifyText(text.trim());
}

// CLI
const cmd = process.argv[2];
if (cmd === "build") {
	await buildCentroids();
} else if (cmd === "classify") {
	const argText = process.argv.slice(3).join(" ");
	if (argText) await classifyText(argText);
	else await classifyFromStdin();
} else {
	console.log(`Usage:
  npm run build
  npm run classify -- "text"
  npm run classify
`);
}
