import * as path from "node:path";
import { getGlobalDaemonRuntimeDir } from "@oh-my-pi/pi-utils";
import { VERSION } from "@oh-my-pi/pi-utils/dirs";

/**
 * Wire types between the parent (`MnemopiEmbedClient`) and the local
 * embeddings subprocess. The parent owns the subprocess lifecycle (graceful
 * work, hard `SIGKILL` on shutdown); the protocol carries no explicit close
 * handshake — once the parent decides to terminate, it signals the OS to reap
 * the child so `onnxruntime-node`'s NAPI finalizer never runs in the main
 * agent address space (it crashes Bun on Windows shutdown — issue #3031, the
 * mnemopi sibling of the tiny-model fix from #1606/#1607). See
 * `embed-client.ts` for the spawn/kill glue.
 */

/** Identifier of the fastembed model the worker should load (e.g. `fast-bge-base-en-v1.5`). */
export type MnemopiEmbedModelId = string;

export type MnemopiEmbedWorkerInbound =
	| { type: "ping"; id: string }
	| { type: "init"; id: string; model: MnemopiEmbedModelId; cacheDir?: string }
	// `embed` always carries the same `model` / `cacheDir` the wrapper was
	// initialized with so a fresh subprocess (after the parent SIGKILLed the
	// previous one but mnemopi still holds the cached `LocalEmbeddingModel`)
	// can lazily reload the model on demand instead of returning
	// "embed before init".
	| { type: "embed"; id: string; model: MnemopiEmbedModelId; cacheDir?: string; texts: string[]; batchSize?: number };

export type MnemopiEmbedWorkerOutbound =
	| { type: "pong"; id: string }
	| { type: "ready"; id: string }
	| { type: "vectors"; id: string; vectors: number[][] }
	| { type: "error"; id: string; error: string }
	// `id` is set when the line was caused by that request, so the shared daemon
	// can forward it to the one client that asked instead of every connection.
	| { type: "log"; id?: string; level: "debug" | "warn" | "error"; msg: string; meta?: Record<string, unknown> };

export interface MnemopiEmbedTransport {
	send(message: MnemopiEmbedWorkerOutbound): void;
	onMessage(handler: (message: MnemopiEmbedWorkerInbound) => void): () => void;
}

/**
 * Daemon-only framing on top of the worker protocol (newline-delimited JSON on
 * a unix socket / named pipe). `hello`/`welcome` is the build handshake: a
 * client on a different build asks the daemon to `shutdown` and respawns it.
 * The per-process child never sees these.
 */
export type MnemopiEmbedDaemonInbound =
	| MnemopiEmbedWorkerInbound
	| { type: "hello"; id: string; build: string }
	// `force` + `stuck` retire a daemon whose embed hung: honoured only when `stuck`
	// names a request of this connection that has been in flight for over half
	// {@link embedRequestTimeoutMs}.
	| { type: "shutdown"; id: string; force?: true; stuck?: string };

export type MnemopiEmbedDaemonOutbound =
	| MnemopiEmbedWorkerOutbound
	| { type: "welcome"; id: string; build: string; pid: number };

/**
 * Daemon build identity. The package version covers releases; bump the revision
 * when this protocol changes without a version bump (dev checkouts).
 */
const PROTOCOL_REV = 2;
export const MNEMOPI_EMBED_BUILD = `${VERSION}+p${PROTOCOL_REV}`;

/**
 * Upper bound on a steady-state embed round-trip; the daemon judges a forced
 * shutdown against it. `MNEMOPI_EMBED_REQUEST_TIMEOUT_MS` overrides it (tests).
 */
export function embedRequestTimeoutMs(): number {
	const requested = Number(process.env.MNEMOPI_EMBED_REQUEST_TIMEOUT_MS);
	return requested > 0 && Number.isFinite(requested) ? requested : 120_000;
}

/**
 * Machine-wide runtime dir (0700): one daemon, one lock, one socket for every
 * omp process of this user. `MNEMOPI_EMBED_DAEMON_DIR` overrides it (tests).
 */
export function embedDaemonDir(): string {
	return process.env.MNEMOPI_EMBED_DAEMON_DIR || getGlobalDaemonRuntimeDir("mnemopi-embed");
}

/** Unix socket under the runtime dir, or a Windows named pipe keyed by that dir (which sits under the user's home). */
export function embedDaemonEndpoint(dir: string): string {
	if (process.platform === "win32") {
		return `\\\\.\\pipe\\omp-mnemopi-embed-${Bun.hash.wyhash(path.resolve(dir)).toString(16).padStart(16, "0")}`;
	}
	return path.join(dir, "embed.sock");
}
