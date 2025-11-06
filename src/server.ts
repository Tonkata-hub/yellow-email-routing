import http from "http";
import * as fs from "fs";
import * as path from "path";
import OpenAI from "openai";
import "dotenv/config";

const PORT = process.env.PORT || 3000;
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const MODEL = "text-embedding-3-small";
const THRESHOLD = 0.4;

type Centroids = Record<string, number[]>;

function normalize(v: number[]): number[] {
	const len = Math.hypot(...v) + 1e-12;
	return v.map((x) => x / len);
}

function cosine(a: number[], b: number[]): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

async function embedTexts(texts: string[]): Promise<number[][]> {
	const res = await client.embeddings.create({ model: MODEL, input: texts });
	return res.data.map((d) => normalize(d.embedding as number[]));
}

async function classifyText(text: string): Promise<any> {
	if (!fs.existsSync("centroids.json")) {
		throw new Error("Centroids not found. Run: npm run build");
	}

	const centroids: Centroids = JSON.parse(fs.readFileSync("centroids.json", "utf-8"));
	const [emb] = await embedTexts([text]);

	const sims = Object.entries(centroids).map(([lab, c]) => [lab, cosine(emb, c)] as const);
	sims.sort((a, b) => b[1] - a[1]);

	const [bestLabel, bestScore] = sims[0];
	const routed = bestScore >= THRESHOLD ? bestLabel : "unclassified";

	return {
		routed,
		bestLabel,
		bestScore: +bestScore.toFixed(3),
		similarities: Object.fromEntries(sims.map(([l, s]) => [l, +s.toFixed(3)]))
	};
}

const server = http.createServer(async (req, res) => {
	const url = new URL(req.url || "/", `http://${req.headers.host}`);

	// CORS headers
	res.setHeader("Access-Control-Allow-Origin", "*");
	res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
	res.setHeader("Access-Control-Allow-Headers", "Content-Type");

	if (req.method === "OPTIONS") {
		res.writeHead(200);
		res.end();
		return;
	}

	if (url.pathname === "/api/classify" && req.method === "POST") {
		let body = "";
		for await (const chunk of req) {
			body += chunk.toString();
		}

		try {
			const { text } = JSON.parse(body);
			if (!text || typeof text !== "string") {
				res.writeHead(400, { "Content-Type": "application/json" });
				res.end(JSON.stringify({ error: "Text is required" }));
				return;
			}

			const result = await classifyText(text);
			res.writeHead(200, { "Content-Type": "application/json" });
			res.end(JSON.stringify(result));
		} catch (error: any) {
			console.error("Classification error:", error);
			res.writeHead(500, { "Content-Type": "application/json" });
			res.end(JSON.stringify({ error: error.message || "Classification failed" }));
		}
		return;
	}

	// Serve HTML
	if (url.pathname === "/" || url.pathname === "/index.html") {
		const htmlPath = path.join(process.cwd(), "public", "index.html");
		if (fs.existsSync(htmlPath)) {
			const html = fs.readFileSync(htmlPath, "utf-8");
			res.writeHead(200, { "Content-Type": "text/html" });
			res.end(html);
		} else {
			res.writeHead(404, { "Content-Type": "text/plain" });
			res.end("404 Not Found");
		}
		return;
	}

	res.writeHead(404, { "Content-Type": "text/plain" });
	res.end("404 Not Found");
});

server.listen(PORT, () => {
	console.log(`🚀 Server running at http://localhost:${PORT}`);
});

