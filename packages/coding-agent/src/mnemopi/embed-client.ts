import * as net from "node:net";
import * as os from "node:os";
import { isFastembedModelCached } from "@oh-my-pi/pi-mnemopi/core/fastembed-model-cache";
import { getFastembedCacheDir, logger } from "@oh-my-pi/pi-utils";
import { MNEMOPI_EMBED_DAEMON_WORKER_ARG } from "../cli/worker-selectors";
import { trackDownload } from "../downloads/activity";
import {
	createUnavailableWorker,
	createWorkerHandle,
	createWorkerSubprocess,
	inferenceWorkerEnv,
	logWorkerMessage,
	resolveWorkerSpawnCmd,
	SMOKE_TEST_TIMEOUT_MS,
	type SpawnedSubprocess,
	smokeTestWorker,
	spawnWorkerOrUnavailable,
	type RefCountedWorkerHandle,
} from "../subprocess/worker-client";
import {
	embedDaemonDir,
	embedDaemonEndpoint,
	embedRequestTimeoutMs,
	MNEMOPI_EMBED_BUILD,
	type MnemopiEmbedDaemonInbound,
	type MnemopiEmbedDaemonOutbound,
	type MnemopiEmbedModelId,
	type MnemopiEmbedWorkerInbound,
	type MnemopiEmbedWorkerOutbound,
} from "./embed-protocol";

/**
 * Parent-side handle for the mnemopi embeddings subprocess. The runtime
 * implementation is a Bun child process so `onnxruntime-node`'s NAPI
 * constructor + finalizer never run inside the main agent address space —
 * those destructors segfault Bun on Windows when mnemopi's local embedding
 * provider loads fastembed in the main process (issue #3031; the mnemopi
 * sibling of the tiny-model fix from #1606 / #1607).
 */
export type MnemopiEmbedWorkerHandle = RefCountedWorkerHandle<MnemopiEmbedWorkerInbound, MnemopiEmbedWorkerOutbound> & {
	/**
	 * Shared daemon only: ask it to exit because request `stuckId` hung (it
	 * checks that claim itself), resolving once the line is flushed. Absent on a private child,
	 * which `terminate()` already SIGKILLs.
	 */
	retire?(stuckId: string): Promise<void>;
};

type PendingRequest =
	| { kind: "init"; model: MnemopiEmbedModelId; resolve: (error: string | undefined) => void }
	| { kind: "embed"; model: MnemopiEmbedModelId; resolve: (vectors: number[][] | Error) => void };

/**
 * Hidden subcommand on the main CLI that boots the mnemopi embeddings worker
 * in the spawned subprocess. Kept in sync with the dispatch in `cli.ts`.
 */
export const MNEMOPI_EMBED_WORKER_ARG = "__omp_worker_mnemopi_embed";

/**
 * Spawn the mnemopi embeddings worker as a subprocess. Exported for tests and
 * the smoke probe; production callers go through {@link createEmbedWorkerHandle}.
 * The child inherits the parent env — fastembed honours `HF_HUB_*`,
 * `HTTPS_PROXY`, etc., and our `loadFastembed()` reads the same `OMP_*`
 * runtime-install knobs the parent uses.
 */
export function createMnemopiEmbedSubprocess(): SpawnedSubprocess<MnemopiEmbedWorkerOutbound> {
	return createWorkerSubprocess<MnemopiEmbedWorkerOutbound>({
		spawnCommand: resolveWorkerSpawnCmd(MNEMOPI_EMBED_WORKER_ARG),
		env: inferenceWorkerEnv(),
		exitLabel: "mnemopi embed subprocess",
	});
}

function wrapSubprocess(spawned: SpawnedSubprocess<MnemopiEmbedWorkerOutbound>): MnemopiEmbedWorkerHandle {
	const { proc } = spawned;
	// Embed keeps its own guarded `proc.send` (neutralizes only the synchronous
	// throw, not the async EPIPE rejection) rather than the shared `safeSend`
	// the other workers use — behaviour preserved verbatim.
	return {
		...createWorkerHandle<MnemopiEmbedWorkerInbound, MnemopiEmbedWorkerOutbound>(spawned, message => {
			try {
				proc.send(message);
			} catch (error) {
				logger.debug("mnemopi-embed: send to subprocess failed", {
					error: error instanceof Error ? error.message : String(error),
				});
			}
		}),
		ref() {
			try {
				proc.ref();
			} catch {
				// Already gone.
			}
		},
		unref() {
			try {
				proc.unref();
			} catch {
				// Already gone.
			}
		},
	};
}

function createUnavailableMnemopiEmbedWorker(error: unknown): MnemopiEmbedWorkerHandle {
	return {
		...createUnavailableWorker<MnemopiEmbedWorkerInbound, MnemopiEmbedWorkerOutbound>(error),
		ref() {},
		unref() {},
	};
}

function spawnMnemopiEmbedProcessWorker(): MnemopiEmbedWorkerHandle {
	return spawnWorkerOrUnavailable(
		() => wrapSubprocess(createMnemopiEmbedSubprocess()),
		createUnavailableMnemopiEmbedWorker,
		"mnemopi embed worker spawn failed; local embeddings disabled",
	);
}

/** Budget for reaching (starting, if needed) the shared daemon: how long to keep retrying while nothing listens. */
const DAEMON_CONNECT_DEADLINE_MS = 5_000;
/** Once connected a daemon exists, though it may be blocked in a synchronous model load: wait this long for its answer. */
const DAEMON_CONNECTED_DEADLINE_MS = 60_000;

export interface EmbedWorkerHandleOptions {
	/** Runtime dir of the daemon; defaults to the machine-wide one. */
	dir?: string;
	/** Build this client expects the daemon to be; a daemon on another build is retired and respawned. */
	build?: string;
	/** Start a daemon process detached. Injectable for tests. */
	spawnDaemon?: () => void;
	/** Private per-process child used when the daemon is disabled or unreachable. */
	spawnChild?: () => MnemopiEmbedWorkerHandle;
	connectDeadlineMs?: number;
}

/**
 * Env the detached daemon gets. It outlives this session and serves every
 * later one, so it must not inherit this session's environment (API keys,
 * tokens, per-project config): only what it needs to find its runtime dir and
 * model, reach the network and load native libraries. The model cache dir
 * travels in each request, not the env. The trailing Windows names are OS
 * basics a process there cannot start without.
 */
const DAEMON_ENV_KEEP =
	/^(PATH|HOME|USER|TMPDIR|XDG_.*|LANG|LC_.*|OMP_NATIVE_LIBRARY_PATH|LD_LIBRARY_PATH|DYLD_.*|HF_HUB_.*|HTTPS?_PROXY|NO_PROXY|MNEMOPI_.*|SystemRoot|windir|TEMP|TMP|USERPROFILE|APPDATA|LOCALAPPDATA|COMSPEC|PATHEXT)$/i;

function spawnEmbedDaemon(): void {
	const spawn = resolveWorkerSpawnCmd(MNEMOPI_EMBED_DAEMON_WORKER_ARG);
	const env: Record<string, string> = {};
	for (const [key, value] of Object.entries(inferenceWorkerEnv())) {
		if (DAEMON_ENV_KEEP.test(key)) env[key] = value;
	}
	// Detached + unref'd: the daemon outlives this session and serves the next ones.
	// cwd is the home dir, never this session's project dir (it would pin that dir
	// and leak into terminal cwd heuristics); `spawn.cwd` is only the package root
	// that a dev checkout's relative `src/cli.ts` entry needs.
	Bun.spawn(spawn.cmd, {
		cwd: spawn.cwd ?? os.homedir(),
		env,
		stdin: "ignore",
		stdout: "ignore",
		stderr: "ignore",
		detached: true,
		windowsHide: true,
	}).unref();
}

function openSocket(endpoint: string): Promise<net.Socket> {
	const { promise, resolve, reject } = Promise.withResolvers<net.Socket>();
	const socket = net.createConnection({ path: endpoint });
	socket.setEncoding("utf8");
	// Permanent: a socket error after connect must never become an uncaught
	// exception; `close` is what the handle reacts to.
	socket.on("error", () => {});
	socket.once("connect", () => {
		socket.removeListener("error", reject);
		resolve(socket);
	});
	socket.once("error", reject);
	return promise;
}

/**
 * First complete JSON line from the daemon plus the bytes buffered behind it,
 * or `undefined` when the socket closes, stays silent for `timeoutMs`, or `signal` aborts.
 */
function nextMessage(
	socket: net.Socket,
	timeoutMs: number,
	signal: AbortSignal,
): Promise<{ message: MnemopiEmbedDaemonOutbound; rest: string } | undefined> {
	const { promise, resolve } = Promise.withResolvers<
		{ message: MnemopiEmbedDaemonOutbound; rest: string } | undefined
	>();
	let buffer = "";
	const done = (result: { message: MnemopiEmbedDaemonOutbound; rest: string } | undefined): void => {
		clearTimeout(timer);
		socket.removeListener("data", onData);
		socket.removeListener("close", onClose);
		signal.removeEventListener("abort", onClose);
		resolve(result);
	};
	const onData = (chunk: string): void => {
		buffer += chunk;
		const newline = buffer.indexOf("\n");
		if (newline < 0) return;
		try {
			done({
				message: JSON.parse(buffer.slice(0, newline)) as MnemopiEmbedDaemonOutbound,
				rest: buffer.slice(newline + 1),
			});
		} catch {
			done(undefined);
		}
	};
	const onClose = (): void => done(undefined);
	const timer = setTimeout(onClose, timeoutMs);
	socket.on("data", onData);
	socket.once("close", onClose);
	signal.addEventListener("abort", onClose, { once: true });
	return promise;
}

function sendLine(socket: net.Socket, message: MnemopiEmbedDaemonInbound): void {
	socket.write(`${JSON.stringify(message)}\n`);
}

/**
 * Connect to the shared daemon, starting it when nothing listens, and check it
 * runs our build. Resolves `undefined` when it cannot be reached or must not
 * be used (wedged, other build still serving other sessions); the caller then
 * falls back to a private child. `rest` is whatever arrived behind the welcome line.
 */
async function connectEmbedDaemon(
	options: Required<Omit<EmbedWorkerHandleOptions, "spawnChild">>,
	signal: AbortSignal,
): Promise<{ socket: net.Socket; rest: string } | undefined> {
	const endpoint = embedDaemonEndpoint(options.dir);
	// Budget for "no daemon yet" (ENOENT/ECONNREFUSED) only; a connected socket is judged separately.
	let deadline = Date.now() + options.connectDeadlineMs;
	let spawned = false;
	let retired = false;
	for (let delay = 25; Date.now() < deadline && !signal.aborted; delay = Math.min(delay * 2, 250)) {
		let socket: net.Socket;
		try {
			socket = await openSocket(endpoint);
		} catch (error) {
			const code = (error as NodeJS.ErrnoException).code;
			if (code !== "ENOENT" && code !== "ECONNREFUSED") return undefined;
			// Nobody listening. Several sessions may race to spawn here; the
			// daemon's lock makes all but one exit at once.
			if (!spawned) {
				spawned = true;
				try {
					options.spawnDaemon();
				} catch (spawnError) {
					logger.warn("mnemopi-embed: spawning the shared daemon failed", {
						error: spawnError instanceof Error ? spawnError.message : String(spawnError),
					});
					return undefined;
				}
			}
			await Bun.sleep(delay);
			continue;
		}
		// A connected socket proves a daemon exists, but it may be inside onnxruntime's
		// synchronous session load and answer only once that returns: keep waiting.
		sendLine(socket, { type: "hello", id: "hello", build: options.build });
		const first = await nextMessage(socket, DAEMON_CONNECTED_DEADLINE_MS, signal);
		const welcome = first?.message;
		if (!first || welcome?.type !== "welcome") {
			socket.destroy();
			return undefined;
		}
		if (welcome.build === options.build) return { socket, rest: first.rest };
		// Other build: retire it once and start ours. A daemon that stays up
		// (refused because other sessions use it, or wedged) means fall back.
		if (retired) {
			socket.destroy();
			return undefined;
		}
		retired = true;
		sendLine(socket, { type: "shutdown", id: "shutdown" });
		const reply = await nextMessage(socket, DAEMON_CONNECTED_DEADLINE_MS, signal);
		socket.destroy();
		if (reply !== undefined) return undefined;
		spawned = false;
		deadline = Date.now() + options.connectDeadlineMs;
	}
	return undefined;
}

/**
 * Worker handle for {@link MnemopiEmbedClient}: a connection to the machine-wide
 * embeddings daemon, so every session shares one loaded model. `send` before
 * the connection is up is queued. A dropped connection surfaces through
 * `onError` like a crashed child, so the client rejects what was in flight and
 * the next call reconnects (respawning the daemon if it died). `terminate()`
 * only closes our connection — the daemon belongs to every session and exits
 * on its own when idle. `MNEMOPI_EMBED_DAEMON=0`, or a daemon that cannot be
 * reached, uses today's private per-process child instead.
 */
export function createEmbedWorkerHandle(options: EmbedWorkerHandleOptions = {}): MnemopiEmbedWorkerHandle {
	const spawnChild = options.spawnChild ?? spawnMnemopiEmbedProcessWorker;
	if (process.env.MNEMOPI_EMBED_DAEMON === "0") return spawnChild();

	const inbound = new Set<(message: MnemopiEmbedWorkerOutbound) => void>();
	const errors = new Set<(error: Error) => void>();
	const queue: MnemopiEmbedWorkerInbound[] = [];
	// Requests sent to the daemon and not yet answered: replayed on a private child if the daemon refuses our model.
	const unanswered = new Map<string, MnemopiEmbedWorkerInbound>();
	const abort = new AbortController();
	let sink: Pick<MnemopiEmbedWorkerHandle, "send" | "terminate" | "ref" | "unref" | "retire"> | undefined;
	let closed = false;
	let refed = false;

	const attach = (target: NonNullable<typeof sink>): void => {
		sink = target;
		if (refed) target.ref();
		else target.unref();
		for (const message of queue.splice(0)) target.send(message);
	};
	const attachChild = (): void => {
		const child = spawnChild();
		child.onMessage(message => {
			for (const handler of inbound) handler(message);
		});
		child.onError(error => {
			for (const handler of errors) handler(error);
		});
		attach(child);
	};
	const attachSocket = (socket: net.Socket, rest: string): void => {
		let buffer = rest;
		let abandoned = false;
		// The daemon refused our (model, cacheDir): it is pinned to another one. Leave it for a
		// private child, which keeps this handle for its lifetime, and replay what was unanswered.
		const abandon = (): void => {
			abandoned = true;
			socket.destroy();
			if (closed) return;
			logger.warn("mnemopi-embed: shared daemon serves another model; using a private embed subprocess");
			queue.push(...unanswered.values());
			unanswered.clear();
			attachChild();
		};
		const drain = (): void => {
			for (let newline = buffer.indexOf("\n"); newline >= 0 && !abandoned; newline = buffer.indexOf("\n")) {
				const line = buffer.slice(0, newline);
				buffer = buffer.slice(newline + 1);
				try {
					const message = JSON.parse(line) as MnemopiEmbedWorkerOutbound;
					if (message.type === "error" && message.code === "pinned") {
						abandon();
						return;
					}
					if (message.type !== "log") unanswered.delete(message.id);
					for (const handler of inbound) handler(message);
				} catch (error) {
					logger.debug("mnemopi-embed: bad daemon line", {
						error: error instanceof Error ? error.message : String(error),
					});
				}
			}
		};
		socket.on("data", (chunk: string) => {
			buffer += chunk;
			drain();
		});
		socket.on("close", () => {
			if (closed || abandoned) return;
			for (const handler of errors) handler(new Error("mnemopi embed daemon connection closed"));
		});
		attach({
			send: message => {
				unanswered.set(message.id, message);
				sendLine(socket, message);
			},
			ref: () => void socket.ref(),
			unref: () => void socket.unref(),
			retire(stuckId) {
				if (socket.destroyed) return Promise.resolve();
				const flushed = Promise.withResolvers<void>();
				socket.once("close", flushed.resolve);
				// `end` flushes the line before closing; a bare `destroy` could drop it.
				socket.end(
					`${JSON.stringify({ type: "shutdown", id: "retire", force: true, stuck: stuckId })}\n`,
					flushed.resolve,
				);
				return flushed.promise;
			},
			async terminate() {
				socket.destroy();
			},
		});
		// Bytes that arrived behind the welcome line.
		drain();
	};

	void connectEmbedDaemon(
		{
			dir: options.dir ?? embedDaemonDir(),
			build: options.build ?? MNEMOPI_EMBED_BUILD,
			spawnDaemon: options.spawnDaemon ?? spawnEmbedDaemon,
			connectDeadlineMs: options.connectDeadlineMs ?? DAEMON_CONNECT_DEADLINE_MS,
		},
		abort.signal,
	)
		.catch(() => undefined)
		.then(connected => {
			if (closed) {
				connected?.socket.destroy();
				return;
			}
			if (connected) return attachSocket(connected.socket, connected.rest);
			logger.warn("mnemopi-embed: shared daemon unavailable; using a private embed subprocess");
			attachChild();
		});

	return {
		send(message) {
			if (sink) sink.send(message);
			else queue.push(message);
		},
		onMessage(handler) {
			inbound.add(handler);
			return () => inbound.delete(handler);
		},
		onError(handler) {
			errors.add(handler);
			return () => errors.delete(handler);
		},
		ref() {
			refed = true;
			sink?.ref();
		},
		unref() {
			refed = false;
			sink?.unref();
		},
		async terminate() {
			closed = true;
			abort.abort();
			queue.length = 0;
			await sink?.terminate();
		},
		async retire(stuckId) {
			await sink?.retire?.(stuckId);
		},
	};
}

/**
 * Per-model wrapper produced by {@link MnemopiEmbedClient.initialize}.
 * `embed()` round-trips one batch of texts through the worker subprocess and
 * yields the resulting vectors in a single asynchronous batch — fastembed's
 * own iterator was emitting batches that we collect on the child side anyway,
 * and serializing per-batch over IPC would not improve throughput.
 */
export interface MnemopiSubprocessEmbeddingModel {
	embed(texts: string[], batchSize?: number): AsyncIterable<number[][]>;
}

/** Race marker for {@link MnemopiEmbedClient.#awaitRequest}. */
const REQUEST_TIMED_OUT = Symbol("mnemopi.embed.timedOut");

export class MnemopiEmbedClient {
	#worker: MnemopiEmbedWorkerHandle | null = null;
	#unsubscribeMessage: (() => void) | null = null;
	#unsubscribeError: (() => void) | null = null;
	#pending = new Map<string, PendingRequest>();
	#nextRequestId = 0;
	#refed = false;
	#spawnWorker: () => MnemopiEmbedWorkerHandle;
	#requestTimeoutMs: number;

	constructor(
		spawnWorker: () => MnemopiEmbedWorkerHandle = createEmbedWorkerHandle,
		requestTimeoutMs: number = embedRequestTimeoutMs(),
	) {
		this.#spawnWorker = spawnWorker;
		this.#requestTimeoutMs = requestTimeoutMs;
	}

	/**
	 * Load the named fastembed model inside the subprocess. Resolves to a
	 * thin wrapper whose `embed()` round-trips through the same worker, or
	 * `null` when the worker cannot init the model (missing peer, native
	 * load failure, etc.). Multiple calls with the same model reuse the
	 * single in-flight worker; calling with a different model loads it on
	 * the child without restarting the process.
	 */
	async initialize(
		model: MnemopiEmbedModelId,
		cacheDir: string | undefined,
	): Promise<MnemopiSubprocessEmbeddingModel | null> {
		// fastembed exposes no byte progress; any missing model file means this
		// init downloads it.
		const resolvedCacheDir = cacheDir ?? getFastembedCacheDir();
		const cached = await isFastembedModelCached(model, resolvedCacheDir);
		const tracker = cached ? undefined : trackDownload(model.replace(/^fast-/, ""), { detail: "downloading" });
		try {
			const worker = this.#ensureWorker();
			const id = String(++this.#nextRequestId);
			const { promise, resolve } = Promise.withResolvers<string | undefined>();
			this.#addPending(id, { kind: "init", model, resolve });
			try {
				// The cacheDir is always explicit: the shared daemon pins on it and may run with another env.
				worker.send({ type: "init", id, model, cacheDir: resolvedCacheDir });
				const error = await promise;
				if (error !== undefined) {
					tracker?.fail(error);
					return null;
				}
				tracker?.done();
			} finally {
				this.#deletePending(id);
			}
		} catch (error) {
			tracker?.fail(error);
			logger.debug("mnemopi-embed: init failed", {
				model,
				error: error instanceof Error ? error.message : String(error),
			});
			return null;
		}
		return { embed: (texts, batchSize) => this.#streamEmbed(model, resolvedCacheDir, texts, batchSize) };
	}

	async terminate(): Promise<void> {
		const worker = this.#worker;
		this.#worker = null;
		this.#unsubscribeMessage?.();
		this.#unsubscribeMessage = null;
		this.#unsubscribeError?.();
		this.#unsubscribeError = null;
		for (const pending of this.#pending.values()) {
			if (pending.kind === "init") pending.resolve("mnemopi embed worker terminated");
			else pending.resolve(new Error("mnemopi embed worker terminated"));
		}
		this.#pending.clear();
		this.#refed = false;
		try {
			await worker?.terminate();
		} catch {
			// Already gone.
		}
	}

	async #embed(
		model: MnemopiEmbedModelId,
		cacheDir: string | undefined,
		texts: string[],
		batchSize: number | undefined,
	): Promise<number[][]> {
		const worker = this.#ensureWorker();
		const id = String(++this.#nextRequestId);
		const { promise, resolve } = Promise.withResolvers<number[][] | Error>();
		this.#addPending(id, { kind: "embed", model, resolve });
		try {
			// Carry the (model, cacheDir) the wrapper was bound to in every
			// embed message: dispose + respawn between two embeds on the same
			// `LocalEmbeddingModel` handle would otherwise hit a fresh
			// worker's "embed before init" guard. Worker `ensureLoaded` is
			// idempotent so steady-state embeds pay no extra cost.
			worker.send({ type: "embed", id, model, cacheDir, texts, batchSize });
			const result = await this.#awaitRequest(promise, id);
			if (result instanceof Error) throw result;
			return result;
		} finally {
			this.#deletePending(id);
		}
	}

	/**
	 * Await one steady-state embed reply, bounded by `#requestTimeoutMs`
	 * ({@link embedRequestTimeoutMs}). Initialization is exempt: bundled installs
	 * may spend minutes bootstrapping the model, and killing that worker can
	 * strand the runtime install lock. The timeout timer is `unref`'d so a
	 * pending request has only the worker reference keeping the parent event
	 * loop alive. On expiry the embed means a hung native runtime (issues #4792,
	 * #7352). A shared daemon is first asked to retire (`retire` proves the hang
	 * by naming this request; otherwise it would stay up and time out every other
	 * session's embeds too), then the worker is reaped via {@link terminate} —
	 * faulting any other in-flight request and letting the next call respawn a
	 * fresh worker — before the request rejects.
	 */
	async #awaitRequest<T>(promise: Promise<T>, id: string): Promise<T> {
		const { promise: timedOut, resolve: fire } = Promise.withResolvers<typeof REQUEST_TIMED_OUT>();
		const timer = setTimeout(() => fire(REQUEST_TIMED_OUT), this.#requestTimeoutMs);
		timer.unref();
		try {
			const winner = await Promise.race([promise, timedOut]);
			if (winner === REQUEST_TIMED_OUT) {
				const worker = this.#worker;
				try {
					await worker?.retire?.(id);
				} catch {
					// Best effort: terminate below drops the connection either way.
				}
				if (this.#worker === worker) void this.terminate();
				throw new Error("mnemopi embed worker request timed out");
			}
			return winner;
		} finally {
			clearTimeout(timer);
		}
	}

	async *#streamEmbed(
		model: MnemopiEmbedModelId,
		cacheDir: string | undefined,
		texts: string[],
		batchSize: number | undefined,
	): AsyncIterable<number[][]> {
		const vectors = await this.#embed(model, cacheDir, texts, batchSize);
		// Mnemopi's `collectMatrix` re-batches via async iteration anyway; yield
		// a single batch carrying the full result so the caller's drain loop
		// behaves identically to the in-process fastembed iterator (one yield
		// per `embed()` call) without paying extra IPC round-trips.
		yield vectors;
	}

	#ensureWorker(): MnemopiEmbedWorkerHandle {
		if (this.#worker) return this.#worker;
		const worker = this.#spawnWorker();
		this.#worker = worker;
		this.#unsubscribeMessage = worker.onMessage(message => this.#handleMessage(message));
		this.#unsubscribeError = worker.onError(error => this.#handleWorkerError(error));
		return worker;
	}

	/** Register a pending request and keep the worker referenced while work is in flight. */
	#addPending(id: string, request: PendingRequest): void {
		this.#pending.set(id, request);
		this.#syncWorkerRef();
	}

	/** Drop a pending request and unref the worker once nothing is in flight. */
	#deletePending(id: string): void {
		if (this.#pending.delete(id)) this.#syncWorkerRef();
	}

	/**
	 * The embeddings subprocess is spawned unref'd so an idle interactive or
	 * daemon session never blocks exit. Keep it referenced only while a request
	 * is pending so short-lived print-mode commands cannot exit before recall
	 * receives the worker response (issue #12067).
	 */
	#syncWorkerRef(): void {
		const worker = this.#worker;
		if (!worker) return;
		const shouldRef = this.#pending.size > 0;
		if (shouldRef === this.#refed) return;
		this.#refed = shouldRef;
		if (shouldRef) worker.ref();
		else worker.unref();
	}

	#handleMessage(message: MnemopiEmbedWorkerOutbound): void {
		if (message.type === "log") {
			logWorkerMessage(message);
			return;
		}
		if (message.type === "pong") return;

		const pending = this.#pending.get(message.id);
		if (!pending) return;
		this.#deletePending(message.id);
		if (message.type === "ready") {
			if (pending.kind === "init") pending.resolve(undefined);
			return;
		}
		if (message.type === "vectors") {
			if (pending.kind === "embed") pending.resolve(message.vectors);
			return;
		}
		logger.debug("mnemopi-embed: worker returned error", { error: message.error });
		if (pending.kind === "init") pending.resolve(message.error);
		else pending.resolve(new Error(message.error));
	}

	#handleWorkerError(error: Error): void {
		logger.warn("mnemopi-embed: worker error", { error: error.message });
		for (const pending of this.#pending.values()) {
			if (pending.kind === "init") pending.resolve(error.message);
			else pending.resolve(error);
		}
		this.#pending.clear();
		void this.terminate();
	}
}

export const mnemopiEmbedClient = new MnemopiEmbedClient();

export async function shutdownMnemopiEmbedClient(): Promise<void> {
	await mnemopiEmbedClient.terminate();
}

export async function smokeTestMnemopiEmbedWorker({
	timeoutMs = SMOKE_TEST_TIMEOUT_MS,
}: {
	timeoutMs?: number;
} = {}): Promise<void> {
	await smokeTestWorker(wrapSubprocess(createMnemopiEmbedSubprocess()), "mnemopi embed worker", timeoutMs);
}
