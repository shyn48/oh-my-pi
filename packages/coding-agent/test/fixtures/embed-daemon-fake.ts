/**
 * Real daemon process for `embed-daemon.test.ts`, with the fastembed loader
 * swapped for a fake so no model or network is needed. Every model load and
 * every embed is appended to `FAKE_EMBED_LOG` (`init <model> pid=<pid>`,
 * `embed <first text>`), which is how the test counts loads across processes.
 * An embed whose first text is `hang` never answers.
 */
import * as fs from "node:fs";
import { startEmbedDaemonFromEnvironment } from "@oh-my-pi/pi-coding-agent/mnemopi/embed-daemon";

const logFile = process.env.FAKE_EMBED_LOG;
if (!logFile) throw new Error("FAKE_EMBED_LOG is required");

await startEmbedDaemonFromEnvironment({
	...(process.env.FAKE_EMBED_BUILD ? { build: process.env.FAKE_EMBED_BUILD } : {}),
	async load(model) {
		fs.appendFileSync(logFile, `init ${model} pid=${process.pid}\n`);
		return {
			async *embed(texts) {
				fs.appendFileSync(logFile, `embed ${texts[0]}\n`);
				if (texts[0] === "hang") await Promise.withResolvers<void>().promise;
				yield texts.map(text => [text.length]);
			},
		};
	},
});
