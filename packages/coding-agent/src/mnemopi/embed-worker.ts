/**
 * Mnemopi local-embeddings worker. Loaded inside the dedicated subprocess
 * spawned by `embed-client.ts` (re-entered through the agent CLI's hidden
 * `__omp_worker_mnemopi_embed` selector), or inside the shared per-machine
 * daemon (`embed-daemon.ts`) that serves many agent processes from one model.
 * The whole point of this module is that `loadFastembed()` — and therefore
 * `onnxruntime-node`'s NAPI constructor + finalizer — only ever runs in that
 * child address space. The parent `SIGKILL`s a per-process child on shutdown
 * (and the daemon kills itself on exit) so the destructor that crashes Bun on
 * Windows shutdown (issue #3031, mnemopi sibling of #1606/#1607) never runs
 * in either process.
 */

import { defaultLocalModelInitializer, type StandardEmbeddingModel } from "@oh-my-pi/pi-mnemopi/core";
import type {
	MnemopiEmbedModelId,
	MnemopiEmbedTransport,
	MnemopiEmbedWorkerInbound,
	MnemopiEmbedWorkerOutbound,
} from "./embed-protocol";

interface LoadedModel {
	model: MnemopiEmbedModelId;
	cacheDir: string | undefined;
	instance: EmbedModelInstance;
}

export interface EmbedModelInstance {
	embed(texts: string[], batchSize?: number): AsyncIterable<number[][]> | Iterable<number[][]>;
}

/** Loads one model; injectable so tests can serve vectors without fastembed/onnxruntime. */
export type EmbedModelLoader = (
	model: MnemopiEmbedModelId,
	cacheDir: string | undefined,
) => Promise<EmbedModelInstance>;

const loadFastembedModel: EmbedModelLoader = (model, cacheDir) =>
	// Route through mnemopi's shared initializer so the worker inherits BOTH
	// cache heals (sidecar re-fetch AND corrupt-model quarantine/retry) —
	// fastembed/onnxruntime still load only in this child address space, the
	// initializer calls loadFastembed() itself.
	// Cast: `model` arrives as a string from the parent (resolved by
	// mnemopi's `fastembedModelName`); the parent only ever passes pre-vetted
	// fast-* identifiers.
	defaultLocalModelInitializer({
		model: model as StandardEmbeddingModel,
		cacheDir,
		showDownloadProgress: false,
	});

/** A pinned handler refused a `(model, cacheDir)` other than the one it serves. */
class PinnedModelError extends Error {}

function errorReply(id: string, error: unknown): MnemopiEmbedWorkerOutbound {
	return {
		type: "error",
		id,
		error: error instanceof Error ? error.message : String(error),
		...(error instanceof PinnedModelError ? { code: "pinned" as const } : {}),
	};
}

export interface EmbedHandlerOptions {
	load?: EmbedModelLoader;
	/**
	 * Serve exactly one `(model, cacheDir)`: a request for another key fails
	 * instead of replacing the loaded model under other clients. The shared
	 * daemon sets this; the per-process child keeps replace-on-change.
	 */
	pinned?: boolean;
}

export interface EmbedHandler {
	/** Answer one request through `send`; resolves once the reply (or error) was sent. */
	handle(message: MnemopiEmbedWorkerInbound, send: (message: MnemopiEmbedWorkerOutbound) => void): Promise<void>;
}

export function createMnemopiEmbedHandler({
	load = loadFastembedModel,
	pinned = false,
}: EmbedHandlerOptions = {}): EmbedHandler {
	let loaded: Promise<LoadedModel> | null = null;
	let loadedKey = "";

	function ensureLoaded(model: MnemopiEmbedModelId, cacheDir: string | undefined): Promise<LoadedModel> {
		const key = `${model}\u0000${cacheDir ?? ""}`;
		if (loaded !== null && loadedKey === key) return loaded;
		if (pinned && loaded !== null) {
			return Promise.reject(
				new PinnedModelError(
					`mnemopi embed daemon is bound to another model/cacheDir; refusing ${model} (cacheDir ${cacheDir ?? "default"})`,
				),
			);
		}
		const loading = load(model, cacheDir)
			.then(instance => ({ model, cacheDir, instance }))
			.catch(error => {
				// Failed loads must not poison the cache — a retry with the same key
				// should re-attempt the load.
				if (loaded === loading) {
					loaded = null;
					loadedKey = "";
				}
				throw error;
			});
		loaded = loading;
		loadedKey = key;
		return loading;
	}

	async function handleEmbed(
		send: (message: MnemopiEmbedWorkerOutbound) => void,
		message: Extract<MnemopiEmbedWorkerInbound, { type: "embed" }>,
	): Promise<void> {
		try {
			// Each `embed` carries the model + cacheDir the wrapper was bound to.
			// `ensureLoaded` is idempotent for the same key, so this is a no-op
			// once the model is in memory — and it transparently re-loads after
			// the parent SIGKILLed the previous subprocess but mnemopi still
			// holds the cached `LocalEmbeddingModel` wrapper from before.
			const { instance } = await ensureLoaded(message.model, message.cacheDir);
			const vectors: number[][] = [];
			const batches = instance.embed([...message.texts], message.batchSize);
			for await (const batch of batches) {
				for (const row of batch) vectors.push(row);
			}
			send({ type: "vectors", id: message.id, vectors });
		} catch (error) {
			send(errorReply(message.id, error));
		}
	}

	async function handleInit(
		send: (message: MnemopiEmbedWorkerOutbound) => void,
		message: Extract<MnemopiEmbedWorkerInbound, { type: "init" }>,
	): Promise<void> {
		try {
			await ensureLoaded(message.model, message.cacheDir);
			send({ type: "ready", id: message.id });
		} catch (error) {
			send(errorReply(message.id, error));
		}
	}

	return {
		async handle(message, send) {
			switch (message.type) {
				case "ping":
					send({ type: "pong", id: message.id });
					return;
				case "init":
					await handleInit(send, message);
					return;
				case "embed":
					await handleEmbed(send, message);
					return;
			}
		},
	};
}

export function startMnemopiEmbedWorker(transport: MnemopiEmbedTransport): void {
	const handler = createMnemopiEmbedHandler();
	transport.onMessage(message => {
		void handler.handle(message, reply => transport.send(reply));
	});
}
