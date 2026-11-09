/**
 * Shared mnemopi embeddings daemon: ONE per machine instead of one ~1 GB
 * onnxruntime child per agent session. Booted through the hidden
 * `__omp_worker_mnemopi_embed_daemon` selector, spawned detached by
 * `embed-client.ts` on first use, and it exits on its own once no client is
 * connected and nothing is in flight for {@link DEFAULT_IDLE_MS}.
 *
 * Wire: newline-delimited JSON on a unix socket / named pipe in the
 * machine-wide runtime dir, carrying the same `ping/init/embed` →
 * `pong/ready/vectors/error` messages the per-process child speaks plus the
 * `hello` → `welcome` build handshake (`embed-protocol.ts`). One daemon serves
 * one `(model, cacheDir)`; any other request is answered with an error and the
 * client falls back to its own child, so a model is never reloaded under
 * another session's feet.
 */

import * as fs from "node:fs";
import * as net from "node:net";
import * as path from "node:path";
import { FileLock } from "@oh-my-pi/pi-natives";
import { postmortem, setProcessName } from "@oh-my-pi/pi-utils";
import {
	embedDaemonDir,
	embedDaemonEndpoint,
	embedRequestTimeoutMs,
	MNEMOPI_EMBED_BUILD,
	type MnemopiEmbedDaemonInbound,
	type MnemopiEmbedDaemonOutbound,
} from "./embed-protocol";
import { createMnemopiEmbedHandler, type EmbedModelLoader } from "./embed-worker";

const DEFAULT_IDLE_MS = 10 * 60_000;
const MAX_LOG_BYTES = 5 * 1024 * 1024;

export interface EmbedDaemonOptions {
	dir: string;
	/** Exit after this long with no connected client and no in-flight request. */
	idleMs: number;
	/** Identity reported in `welcome`; clients on another build ask us to shut down. */
	build: string;
	/** Model loader override for tests. */
	load?: EmbedModelLoader;
	/** Embed timeout clients apply; a forced shutdown needs a request older than half of it. Defaults to the shared value. */
	requestTimeoutMs?: number;
}

/** Append-only log in the runtime dir; truncated when it outgrows {@link MAX_LOG_BYTES}. Never throws. */
function createDaemonLog(file: string): (level: string, msg: string, meta?: Record<string, unknown>) => void {
	let bytes = 0;
	try {
		bytes = fs.statSync(file).size;
	} catch {
		// First run.
	}
	return (level, msg, meta) => {
		const line = `${new Date().toISOString()} ${level} ${msg}${meta ? ` ${JSON.stringify(meta)}` : ""}\n`;
		try {
			if (bytes + line.length > MAX_LOG_BYTES) {
				fs.writeFileSync(file, "", { mode: 0o600 });
				bytes = 0;
			}
			fs.appendFileSync(file, line, { mode: 0o600 });
			bytes += line.length;
		} catch {
			// Logging must never take the daemon down.
		}
	};
}

/**
 * Serve until idle or told to shut down. Resolves `false` immediately when
 * another daemon owns the lock (the OS drops the lock when its owner dies, so
 * a held lock means a live daemon), `true` after serving.
 */
export async function runEmbedDaemon(options: EmbedDaemonOptions): Promise<boolean> {
	const { dir, idleMs, build } = options;
	const requestTimeoutMs = options.requestTimeoutMs ?? embedRequestTimeoutMs();
	fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
	const lock = FileLock.tryAcquire(path.join(dir, "embed.lock"));
	if (!lock.acquired) return false;

	const endpoint = embedDaemonEndpoint(dir);
	const log = createDaemonLog(path.join(dir, "embed.log"));
	const handler = createMnemopiEmbedHandler({ load: options.load, pinned: true });
	const clients = new Set<net.Socket>();
	const finished = Promise.withResolvers<void>();
	let inflight = 0;
	let idleTimer: NodeJS.Timeout | undefined;
	let stopping = false;

	const stop = (reason: string): void => {
		if (stopping) return;
		stopping = true;
		log("info", `stopping: ${reason}`);
		finished.resolve();
	};
	// Idle clock runs only while nobody is connected and nothing is in flight:
	// the model stays loaded for as long as any session holds a connection.
	const scheduleIdle = (): void => {
		clearTimeout(idleTimer);
		idleTimer = undefined;
		if (stopping || clients.size > 0 || inflight > 0) return;
		idleTimer = setTimeout(() => stop("idle"), idleMs);
	};

	const server = net.createServer(socket => {
		clients.add(socket);
		// Start time of each request this connection has in flight, by request id
		// (ids are per-client counters, so they are only unique per connection).
		const started = new Map<string, number>();
		scheduleIdle();
		socket.setEncoding("utf8");
		const send = (message: MnemopiEmbedDaemonOutbound): void => {
			// A log line without a request id belongs to the daemon, not to
			// whichever client happens to be connected.
			if (message.type === "log" && message.id === undefined) {
				log(message.level, message.msg, message.meta);
				return;
			}
			if (!socket.destroyed) socket.write(`${JSON.stringify(message)}\n`);
		};
		const dispatch = (message: MnemopiEmbedDaemonInbound): void => {
			switch (message.type) {
				case "hello":
					send({ type: "welcome", id: message.id, build, pid: process.pid });
					return;
				case "shutdown": {
					if (message.force) {
						// Proof of a hang: the named request has been in flight for over
						// half the client timeout. Then no other session can be served by
						// this daemon either, however many are connected.
						const since = message.stuck === undefined ? undefined : started.get(message.stuck);
						if (since !== undefined && Date.now() - since > requestTimeoutMs / 2) {
							stop(`embed request ${message.stuck} hung`);
						} else {
							send({ type: "error", id: message.id, error: "forced shutdown refused: no such hung request" });
						}
						return;
					}
					// A mismatched client may only retire a daemon nobody else uses;
					// otherwise mixed-version sessions would kill each other's daemon in a loop.
					if (clients.size === 1) stop("shutdown requested by a client on another build");
					else send({ type: "error", id: message.id, error: "mnemopi embed daemon is in use by other sessions" });
					return;
				}
				default:
					inflight++;
					started.set(message.id, Date.now());
					void handler.handle(message, send).finally(() => {
						inflight--;
						started.delete(message.id);
						scheduleIdle();
					});
			}
		};
		let buffer = "";
		socket.on("data", (chunk: string) => {
			buffer += chunk;
			for (let newline = buffer.indexOf("\n"); newline >= 0; newline = buffer.indexOf("\n")) {
				const line = buffer.slice(0, newline);
				buffer = buffer.slice(newline + 1);
				if (!line) continue;
				try {
					dispatch(JSON.parse(line) as MnemopiEmbedDaemonInbound);
				} catch (error) {
					log("warn", "dropped malformed client line", {
						error: error instanceof Error ? error.message : String(error),
					});
				}
			}
		});
		socket.on("error", () => {
			// `close` follows and does the bookkeeping.
		});
		socket.on("close", () => {
			clients.delete(socket);
			scheduleIdle();
		});
	});

	const cancelCleanup = postmortem.register("mnemopi-embed-daemon", () => stop("process signal"));
	try {
		// We hold the lock, so any socket file is a corpse from a dead daemon.
		if (process.platform !== "win32") fs.rmSync(endpoint, { force: true });
		const listening = Promise.withResolvers<void>();
		server.once("listening", listening.resolve);
		server.once("error", listening.reject);
		server.listen(endpoint);
		await listening.promise;
		if (process.platform !== "win32") fs.chmodSync(endpoint, 0o600);
		log("info", "listening", { pid: process.pid, build, idleMs });
		scheduleIdle();
		await finished.promise;
	} finally {
		cancelCleanup();
		clearTimeout(idleTimer);
		// Order matters: drop the socket file and the lock BEFORE disconnecting
		// clients, so a client that reacts to the close by respawning us finds a
		// free lock instead of losing to a daemon that is already leaving.
		if (process.platform !== "win32") fs.rmSync(endpoint, { force: true });
		lock.release();
		for (const socket of clients) socket.destroy();
		server.close();
	}
	return true;
}

export async function startEmbedDaemonFromEnvironment(options: Partial<EmbedDaemonOptions> = {}): Promise<void> {
	setProcessName("omp embed daemon");
	const requestedIdle = Number(process.env.MNEMOPI_EMBED_DAEMON_IDLE_MS);
	const idleMs =
		options.idleMs ??
		(process.env.MNEMOPI_EMBED_DAEMON_IDLE_MS && Number.isFinite(requestedIdle) && requestedIdle >= 0
			? requestedIdle
			: DEFAULT_IDLE_MS);
	const served = await runEmbedDaemon({
		dir: embedDaemonDir(),
		idleMs,
		build: MNEMOPI_EMBED_BUILD,
		...options,
	});
	// We loaded onnxruntime: die by SIGKILL so its NAPI finalizer never runs
	// (issue #3031 — it crashes Bun on Windows at normal exit).
	if (served) process.kill(process.pid, "SIGKILL");
}
