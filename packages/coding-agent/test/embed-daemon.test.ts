/**
 * The mnemopi embeddings worker runs as ONE shared daemon per machine instead
 * of one ~1 GB onnxruntime child per agent session. These tests drive real
 * daemon processes (`fixtures/embed-daemon-fake.ts`, fastembed swapped for a
 * fake loader) through the real client over a real unix socket, in a throwaway
 * runtime dir. Real processes mean real clocks: the idle test sleeps past a
 * 200 ms idle window, everything else polls for the condition it awaits.
 */
import { afterEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import {
	createEmbedWorkerHandle,
	MnemopiEmbedClient,
	type MnemopiEmbedWorkerHandle,
	type MnemopiSubprocessEmbeddingModel,
} from "@oh-my-pi/pi-coding-agent/mnemopi/embed-client";
import { embedDaemonEndpoint } from "@oh-my-pi/pi-coding-agent/mnemopi/embed-protocol";
import type { Subprocess } from "bun";

const FIXTURE = path.join(import.meta.dir, "fixtures", "embed-daemon-fake.ts");
const BUILD = "test-build";
const MODEL = "fast-fake-model";

const procs: Subprocess[] = [];
const clients: MnemopiEmbedClient[] = [];
const dirs: string[] = [];

afterEach(async () => {
	for (const client of clients.splice(0)) await client.terminate();
	for (const proc of procs.splice(0)) {
		proc.kill("SIGKILL");
		await proc.exited;
	}
	for (const dir of dirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
});

/** Short base so `<dir>/embed.sock` fits `sun_path` even where `os.tmpdir()` is a long sandbox path. */
const TMP_BASE = process.platform === "win32" ? os.tmpdir() : "/tmp";

function makeDir(): string {
	const dir = fs.mkdtempSync(path.join(TMP_BASE, "omp-embed-"));
	dirs.push(dir);
	return dir;
}

function startDaemon(dir: string, env: Record<string, string> = {}): Subprocess {
	const proc = Bun.spawn([process.execPath, FIXTURE], {
		env: {
			...process.env,
			MNEMOPI_EMBED_DAEMON_DIR: dir,
			MNEMOPI_EMBED_DAEMON_IDLE_MS: "60000",
			FAKE_EMBED_LOG: path.join(dir, "fake.log"),
			FAKE_EMBED_BUILD: BUILD,
			...env,
		},
		stdin: "ignore",
		stdout: "ignore",
		stderr: "ignore",
	});
	procs.push(proc);
	return proc;
}

/** Client whose daemon spawner starts the fixture and counts how often it was asked to. */
function makeClient(
	dir: string,
	env: Record<string, string> = {},
	requestTimeoutMs?: number,
	spawnChild: () => MnemopiEmbedWorkerHandle = () => {
		throw new Error("per-process child must not be used");
	},
): { client: MnemopiEmbedClient; spawns: () => number } {
	let spawns = 0;
	const client = new MnemopiEmbedClient(
		() =>
			createEmbedWorkerHandle({
				dir,
				build: BUILD,
				spawnDaemon: () => {
					spawns += 1;
					startDaemon(dir, env);
				},
				spawnChild,
			}),
		requestTimeoutMs,
	);
	clients.push(client);
	return { client, spawns: () => spawns };
}

async function embedOnce(wrapper: MnemopiSubprocessEmbeddingModel, text: string): Promise<number[][]> {
	const out: number[][] = [];
	for await (const batch of wrapper.embed([text])) out.push(...batch);
	return out;
}

function fakeLog(dir: string): string[] {
	try {
		return fs.readFileSync(path.join(dir, "fake.log"), "utf8").split("\n").filter(Boolean);
	} catch {
		return [];
	}
}

async function waitFor(predicate: () => boolean, timeoutMs = 10_000): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	while (!predicate()) {
		if (Date.now() > deadline) throw new Error("timed out waiting for condition");
		await Bun.sleep(20);
	}
}

const exitedOrTimeout = (proc: Subprocess, ms: number): Promise<number | "timeout"> =>
	Promise.race([proc.exited, Bun.sleep(ms).then(() => "timeout" as const)]);

describe("shared mnemopi embed daemon", () => {
	it("serves two clients from one daemon and one model load", async () => {
		const dir = makeDir();
		const a = makeClient(dir);
		const b = makeClient(dir);

		const wrapperA = await a.client.initialize(MODEL, undefined);
		const wrapperB = await b.client.initialize(MODEL, undefined);
		expect(wrapperA).not.toBeNull();
		expect(wrapperB).not.toBeNull();

		expect(await embedOnce(wrapperA!, "hello")).toEqual([[5]]);
		expect(await embedOnce(wrapperB!, "hi")).toEqual([[2]]);

		// The second client found the daemon already up: it never spawned one,
		// and the model was loaded exactly once for both.
		expect(a.spawns()).toBe(1);
		expect(b.spawns()).toBe(0);
		expect(fakeLog(dir).filter(line => line.startsWith("init "))).toHaveLength(1);
		expect(procs).toHaveLength(1);
	}, 30_000);

	it("a second daemon process loses the lock and exits 0 while the first keeps serving", async () => {
		const dir = makeDir();
		const { client } = makeClient(dir);
		const wrapper = await client.initialize(MODEL, undefined);
		expect(wrapper).not.toBeNull();

		const loser = startDaemon(dir);
		expect(await exitedOrTimeout(loser, 10_000)).toBe(0);
		expect(await embedOnce(wrapper!, "still-here")).toEqual([[10]]);
	}, 30_000);

	it("refuses a different model and serves that client from a private child instead", async () => {
		const dir = makeDir();
		const a = makeClient(dir);
		let childSpawns = 0;
		const b = makeClient(dir, {}, undefined, () => {
			childSpawns += 1;
			return fakeChild();
		});
		const wrapperA = await a.client.initialize(MODEL, undefined);
		expect(wrapperA).not.toBeNull();

		const wrapperB = await b.client.initialize("fast-other-model", undefined);
		expect(wrapperB).not.toBeNull();
		// The vectors come from the private child (the fake daemon would answer the same), and
		// the daemon neither reloaded nor was left with a dropped-connection error.
		expect(await embedOnce(wrapperB!, "abcd")).toEqual([[4]]);
		expect(childSpawns).toBe(1);
		expect(await embedOnce(wrapperA!, "abc")).toEqual([[3]]);
		expect(fakeLog(dir).filter(line => line.startsWith("init "))).toHaveLength(1);
		expect(fakeLog(dir).some(line => line.startsWith("embed abcd"))).toBe(false);
	}, 30_000);

	it("exits when idle, but keeps the model while a client stays connected", async () => {
		const dir = makeDir();
		const { client } = makeClient(dir, { MNEMOPI_EMBED_DAEMON_IDLE_MS: "200" });
		const wrapper = await client.initialize(MODEL, undefined);
		expect(wrapper).not.toBeNull();
		const daemon = procs[0]!;

		await Bun.sleep(700);
		expect(daemon.exitCode).toBeNull();
		expect(await embedOnce(wrapper!, "x")).toEqual([[1]]);
		expect(fakeLog(dir).filter(line => line.startsWith("init "))).toHaveLength(1);

		await client.terminate();
		expect(await exitedOrTimeout(daemon, 10_000)).not.toBe("timeout");
		expect(fs.existsSync(path.join(dir, "embed.sock"))).toBe(false);
	}, 30_000);

	it("retires a daemon on another build and respawns one on ours", async () => {
		const dir = makeDir();
		const old = startDaemon(dir, { FAKE_EMBED_BUILD: "old-build" });
		await waitFor(() => fs.existsSync(path.join(dir, "embed.sock")));

		const { client, spawns } = makeClient(dir);
		const wrapper = await client.initialize(MODEL, undefined);
		expect(wrapper).not.toBeNull();

		expect(await exitedOrTimeout(old, 10_000)).not.toBe("timeout");
		expect(spawns()).toBe(1);
		expect(await embedOnce(wrapper!, "fresh")).toEqual([[5]]);
	}, 30_000);

	it("rejects the in-flight request when the daemon dies and the next call respawns it", async () => {
		const dir = makeDir();
		const { client, spawns } = makeClient(dir);
		const wrapper = await client.initialize(MODEL, undefined);
		expect(wrapper).not.toBeNull();
		const pid = Number(/pid=(\d+)/.exec(fakeLog(dir)[0]!)![1]);

		const inFlight = embedOnce(wrapper!, "hang");
		const settled = inFlight.then(
			() => "resolved",
			() => "rejected",
		);
		await waitFor(() => fakeLog(dir).includes("embed hang"));
		process.kill(pid, "SIGKILL");
		expect(await settled).toBe("rejected");

		// The dead daemon leaves a stale socket file behind; the respawned one must replace it.
		expect(await embedOnce(wrapper!, "again")).toEqual([[5]]);
		expect(spawns()).toBe(2);
		expect(fakeLog(dir).filter(line => line.startsWith("init "))).toHaveLength(2);
	}, 30_000);

	it("retires a daemon whose embed hangs so the next session gets a fresh one", async () => {
		const dir = makeDir();
		// The daemon judges the hang against this timeout (half of it must have elapsed);
		// its own wedge budget stays above the clients' 3 s so the client-side retire path runs.
		const env = { MNEMOPI_EMBED_REQUEST_TIMEOUT_MS: "4000" };
		const a = makeClient(dir, env, 3000);
		const b = makeClient(dir, env, 3000);
		const wrapperA = await a.client.initialize(MODEL, undefined);
		const wrapperB = await b.client.initialize(MODEL, undefined);
		expect(wrapperA).not.toBeNull();
		expect(wrapperB).not.toBeNull();
		const wedged = procs[0]!;

		// Two clients are connected, so only the hang itself can justify the shutdown.
		await expect(embedOnce(wrapperA!, "hang")).rejects.toThrow("timed out");
		expect(await exitedOrTimeout(wedged, 10_000)).not.toBe("timeout");
		expect(fs.existsSync(path.join(dir, "embed.sock"))).toBe(false);

		// Both sessions recover on their next call against one fresh daemon.
		expect(await embedOnce(wrapperA!, "again")).toEqual([[5]]);
		expect(await embedOnce(wrapperB!, "bee")).toEqual([[3]]);
		expect(fakeLog(dir).filter(line => line.startsWith("init "))).toHaveLength(2);
	}, 30_000);

	it("a forced shutdown naming an unknown request is refused", async () => {
		const dir = makeDir();
		const daemon = startDaemon(dir);
		await waitFor(() => fs.existsSync(path.join(dir, "embed.sock")));
		const socket = net.createConnection({ path: path.join(dir, "embed.sock") });
		socket.setEncoding("utf8");
		const lines: string[] = [];
		socket.on("data", (chunk: string) => lines.push(...chunk.split("\n").filter(Boolean)));
		const send = (message: object): void => void socket.write(`${JSON.stringify(message)}\n`);
		const reply = async (id: string): Promise<{ type: string }> => {
			await waitFor(() => lines.some(line => JSON.parse(line).id === id));
			return JSON.parse(lines.find(line => JSON.parse(line).id === id)!);
		};
		try {
			send({ type: "init", id: "i", model: MODEL });
			expect((await reply("i")).type).toBe("ready");

			send({ type: "shutdown", id: "s1", force: true, stuck: "nope" });
			expect((await reply("s1")).type).toBe("error");

			// A real request that is not yet old enough (default timeout is 120 s) is no proof either.
			send({ type: "embed", id: "e1", model: MODEL, texts: ["hang"] });
			await waitFor(() => fakeLog(dir).includes("embed hang"));
			send({ type: "shutdown", id: "s2", force: true, stuck: "e1" });
			expect((await reply("s2")).type).toBe("error");

			expect(await exitedOrTimeout(daemon, 300)).toBe("timeout");
		} finally {
			socket.destroy();
		}
	}, 30_000);

	it("MNEMOPI_EMBED_DAEMON=0 uses the per-process child and never touches the daemon", async () => {
		const previous = process.env.MNEMOPI_EMBED_DAEMON;
		process.env.MNEMOPI_EMBED_DAEMON = "0";
		try {
			let childSpawns = 0;
			const child = (): MnemopiEmbedWorkerHandle => {
				childSpawns += 1;
				return fakeChild();
			};
			const client = new MnemopiEmbedClient(() =>
				createEmbedWorkerHandle({
					dir: makeDir(),
					spawnChild: child,
					spawnDaemon: () => {
						throw new Error("daemon must not be spawned");
					},
				}),
			);
			clients.push(client);
			const wrapper = await client.initialize(MODEL, undefined);
			expect(wrapper).not.toBeNull();
			expect(await embedOnce(wrapper!, "abcd")).toEqual([[4]]);
			expect(childSpawns).toBe(1);
		} finally {
			if (previous === undefined) delete process.env.MNEMOPI_EMBED_DAEMON;
			else process.env.MNEMOPI_EMBED_DAEMON = previous;
		}
	}, 15_000);

	it("falls back to the per-process child when the daemon never comes up", async () => {
		let childSpawns = 0;
		const client = new MnemopiEmbedClient(() =>
			createEmbedWorkerHandle({
				dir: makeDir(),
				spawnDaemon: () => {},
				connectDeadlineMs: 300,
				spawnChild: () => {
					childSpawns += 1;
					return fakeChild();
				},
			}),
		);
		clients.push(client);
		const wrapper = await client.initialize(MODEL, undefined);
		expect(wrapper).not.toBeNull();
		expect(await embedOnce(wrapper!, "ab")).toEqual([[2]]);
		expect(childSpawns).toBe(1);
	}, 15_000);

	it("waits out a daemon blocked in a synchronous model load instead of using a private child", async () => {
		const dir = makeDir();
		const env = { FAKE_EMBED_LOAD_BLOCK_MS: "3000" };
		const a = makeClient(dir, env);
		const b = makeClient(dir, env);

		const initA = a.client.initialize(MODEL, undefined);
		// The daemon is now frozen inside the load; the second client connects into that silence.
		await waitFor(() => fakeLog(dir).some(line => line.startsWith("init ")));
		const initB = b.client.initialize(MODEL, undefined);
		const [wrapperA, wrapperB] = await Promise.all([initA, initB]);
		expect(wrapperA).not.toBeNull();
		expect(wrapperB).not.toBeNull();

		expect(await embedOnce(wrapperA!, "hello")).toEqual([[5]]);
		expect(await embedOnce(wrapperB!, "hi")).toEqual([[2]]);
		// A spawnChild call would have thrown inside makeClient; one daemon, one load.
		expect(a.spawns() + b.spawns()).toBe(1);
		expect(procs).toHaveLength(1);
		expect(fakeLog(dir).filter(line => line.startsWith("init "))).toHaveLength(1);
	}, 30_000);

	it("an old-build daemon honours shutdown from two new-build clients that arrive together", async () => {
		const dir = makeDir();
		const old = startDaemon(dir, { FAKE_EMBED_BUILD: "old-build" });
		await waitFor(() => fs.existsSync(path.join(dir, "embed.sock")));
		const a = makeClient(dir);
		const b = makeClient(dir);

		const [wrapperA, wrapperB] = await Promise.all([
			a.client.initialize(MODEL, undefined),
			b.client.initialize(MODEL, undefined),
		]);
		expect(wrapperA).not.toBeNull();
		expect(wrapperB).not.toBeNull();
		expect(await exitedOrTimeout(old, 10_000)).not.toBe("timeout");

		expect(await embedOnce(wrapperA!, "hello")).toEqual([[5]]);
		expect(await embedOnce(wrapperB!, "hi")).toEqual([[2]]);
		// One new daemon serves both: a single load (a racing second spawn loses the lock).
		expect(fakeLog(dir).filter(line => line.startsWith("init "))).toHaveLength(1);
	}, 30_000);

	it("moves the socket to the temp dir when the runtime dir path is too long", async () => {
		const dir = path.join(makeDir(), "d".repeat(110));
		fs.mkdirSync(dir);
		// The fallback socket sits in os.tmpdir(), which can itself be long (sandboxes); pin it short.
		const previousTmp = process.env.TMPDIR;
		process.env.TMPDIR = TMP_BASE;
		try {
			const endpoint = embedDaemonEndpoint(dir);
			expect(endpoint).not.toContain(dir);
			expect(path.basename(endpoint)).toStartWith("omp-mnemopi-embed-");
			dirs.push(endpoint);

			const { client } = makeClient(dir);
			const wrapper = await client.initialize(MODEL, undefined);
			expect(wrapper).not.toBeNull();
			expect(await embedOnce(wrapper!, "long")).toEqual([[4]]);
			expect(fs.existsSync(endpoint)).toBe(true);
			expect(fs.existsSync(path.join(dir, "embed.lock"))).toBe(true);
		} finally {
			if (previousTmp === undefined) delete process.env.TMPDIR;
			else process.env.TMPDIR = previousTmp;
		}
	}, 30_000);

	it("retires itself when an embed outlives the request budget, with no client action", async () => {
		const dir = makeDir();
		// The client never times out here: only the daemon's own budget can end this.
		const { client } = makeClient(dir, { MNEMOPI_EMBED_REQUEST_TIMEOUT_MS: "500" }, 60_000);
		const wrapper = await client.initialize(MODEL, undefined);
		expect(wrapper).not.toBeNull();
		const daemon = procs[0]!;

		const inFlight = embedOnce(wrapper!, "hang").then(
			() => "resolved",
			() => "rejected",
		);
		expect(await exitedOrTimeout(daemon, 10_000)).not.toBe("timeout");
		expect(await inFlight).toBe("rejected");
	}, 30_000);
});

/** In-process stand-in for the per-process child: vectors are `[text.length]`. */
function fakeChild(): MnemopiEmbedWorkerHandle {
	const handlers = new Set<Parameters<MnemopiEmbedWorkerHandle["onMessage"]>[0]>();
	return {
		send(message) {
			queueMicrotask(() => {
				for (const handler of handlers) {
					if (message.type === "init") handler({ type: "ready", id: message.id });
					else if (message.type === "embed") {
						handler({ type: "vectors", id: message.id, vectors: message.texts.map(text => [text.length]) });
					}
				}
			});
		},
		onMessage(handler) {
			handlers.add(handler);
			return () => handlers.delete(handler);
		},
		onError: () => () => {},
		ref() {},
		unref() {},
		async terminate() {},
	};
}
