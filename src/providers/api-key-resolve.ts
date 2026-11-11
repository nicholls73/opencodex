import { publishCredentialSync } from "../local-credentials/generation";
/**
 * Read-path for provider key material: env references, keychain references, or the literal
 * value.
 *
 * Leaf module on purpose: reasoning-metadata.ts imports resolveProviderApiKey() here so a
 * learned-refusal identity can hash the same credential the wire sends. key-store.ts needs
 * the ../config barrel for the write path (saveConfigPreservingClaudeCode), and importing it
 * from reasoning-metadata would close the cycle the reasoning-metadata header warns about.
 * resolveEnvValue comes from ../config/proxy-env for the same reason.
 *
 * `config.json` keeps only a reference (`keychain:<provider>` for the active key,
 * `keychain:<provider>/<pool id>` for pool entries); the secret lives in the OS credential store
 * under one service name. Reads are synchronous on purpose: `routedProviderConfig` and the
 * quota/compaction/catalog callers are all sync, and `@napi-rs/keyring` ships a sync `Entry`.
 *
 * Policy: a reference that cannot be resolved fails closed (no key) and is warned once per
 * account; nothing ever rewrites plaintext into config or its backups.
 */
import { invalidateDecisionKeyQuotas, observeDecisionKeyCredential } from "../lib/decision-quota-sink";
import { resolveEnvValue } from "../config/proxy-env";
import { loadKeyringBinding } from "../lib/keyring-native";
import { isGcpCredentialMaterial } from "../lib/gcp-credential-input";
import { CANONICAL_JEV_DECISION_PROVIDER, isOpenAIDecisionsEndpoint } from "../combos/jev-decision-contract";
import type { OcxProviderConfig } from "../types";

export const KEYCHAIN_REFERENCE_PREFIX = "keychain:";
const SELF_HOSTED_RESERVED_ENV_KEYS = new Set(["TYPESAFE_API_KEY", "JEV_API_KEY"]);
/** Stored GCP JSON uses the same OS service, but must never be restored as a literal API key. */
export const GCP_CREDENTIAL_MARKER_PREFIX = "gcp-sa:";
export const PROVIDER_KEYCHAIN_SERVICE = "opencodex.provider-api-key.v1";

export class GcpCredentialAsApiKeyError extends Error {
  constructor() {
    super("GCP credential JSON cannot be used as a literal API key; select a gcp-sa reference");
    this.name = "GcpCredentialAsApiKeyError";
  }
}

/** Config presence only: do not resolve a secret just to report whether a key slot is filled. */
export function hasConfiguredProviderApiKey(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

/** Only native OpenAI Decisions needs an existing reference to resolve before setup is usable. */
export function hasUsableProviderApiKey(name: string, provider: OcxProviderConfig): boolean {
  if (!hasConfiguredProviderApiKey(provider.apiKey)) return false;
  const nativeOpenAI = name !== CANONICAL_JEV_DECISION_PROVIDER
    && provider.adapter === "jev-decision" && isOpenAIDecisionsEndpoint(provider.baseUrl);
  if (!nativeOpenAI) return true;
  if (!selfHostedDecisionApiKeyReferenceAllowed(name, provider.apiKey)) return false;
  try {
    return Boolean(resolveProviderApiKeyQuietly(provider.apiKey)?.trim());
  } catch (error) {
    if (error instanceof GcpCredentialAsApiKeyError) return false;
    throw error;
  }
}

/** Native OpenAI Decisions uses its own key even when the row declares another auth mode. */
export function isKeyAuthProvider(provider: OcxProviderConfig, name?: string): boolean {
  return (provider.authMode !== "oauth" && provider.authMode !== "forward")
    || (name !== undefined && name !== CANONICAL_JEV_DECISION_PROVIDER
      && provider.adapter === "jev-decision" && isOpenAIDecisionsEndpoint(provider.baseUrl));
}

export interface ProviderKeychainEntry {
  getPassword(): string | null;
  setPassword(password: string): void;
  deletePassword(): boolean;
}

export type ProviderKeychainEntryFactory = (service: string, account: string) => ProviderKeychainEntry;

function defaultEntryFactory(service: string, account: string): ProviderKeychainEntry {
  const { Entry } = loadKeyringBinding() as { Entry: new (s: string, a: string) => ProviderKeychainEntry };
  return new Entry(service, account);
}

let entryFactory: ProviderKeychainEntryFactory = defaultEntryFactory;
const resolvedCache = new Map<string, string>();
const warnedAccounts = new Set<string>();
let resolutionGeneration = 0;

/** Advances whenever cached secrets may have changed, so derived snapshots can rebuild. */
export function providerKeyResolutionGeneration(): number {
  return resolutionGeneration;
}

/** Test seam: swap the OS entry for an in-memory one and drop caches. */
export function setProviderKeychainEntryFactoryForTests(factory: ProviderKeychainEntryFactory | null): void {
  entryFactory = factory ?? defaultEntryFactory;
  invalidateDecisionKeyQuotas();
  resolvedCache.clear();
  resolutionGeneration++;
  warnedAccounts.clear();
}

/** Write-path seam: a store/restore mutated secrets, so cached reads and warnings are stale. */
export function invalidateResolvedProviderKeyCache(): void {
  invalidateDecisionKeyQuotas();
  resolvedCache.clear();
  resolutionGeneration++;
  warnedAccounts.clear();
}

export function isKeychainReference(value: string | undefined): value is string {
  return typeof value === "string" && value.startsWith(KEYCHAIN_REFERENCE_PREFIX) && value.length > KEYCHAIN_REFERENCE_PREFIX.length;
}

export function keychainAccount(reference: string): string {
  return reference.slice(KEYCHAIN_REFERENCE_PREFIX.length);
}

/**
 * A reference belongs to `name` only when its account is that provider's own active account
 * or one of its pool accounts. `storeProviderKeyInKeychain` writes exactly those two shapes,
 * so anything else in a provider's config names another provider's secret.
 */
export function keychainReferenceBelongsToProvider(reference: string, name: string): boolean {
  const account = keychainAccount(reference);
  return account === name || account.startsWith(`${name}/`);
}

/** Self-hosted decision rows cannot borrow TypeSafe env keys or another provider's keychain entry. */
export function selfHostedDecisionApiKeyReferenceAllowed(name: string, value: string): boolean {
  const braced = /^\$\{(\w+)\}$/.exec(value);
  const envName = braced?.[1] ?? (value.startsWith("$") ? value.slice(1) : undefined);
  return !(envName !== undefined && SELF_HOSTED_RESERVED_ENV_KEYS.has(envName))
    && (!isKeychainReference(value) || keychainReferenceBelongsToProvider(value, name));
}

/** Entry for `account` under the provider-key service; read and write paths share the factory. */
export function providerKeychainEntry(account: string): ProviderKeychainEntry {
  const entry = entryFactory(PROVIDER_KEYCHAIN_SERVICE, account);
  // A failed or unverifiable write keeps its account counted as a local credential.
  const absent = (): readonly string[] => {
    try { return entry.getPassword() === null ? [account] : []; } catch { return []; }
  };
  return {
    getPassword: () => entry.getPassword(),
    setPassword: password => publishCredentialSync(() => entry.setPassword(password), {
      accounts: [account], absent: completed => completed ? [] : absent(),
    }),
    // Registered before the attempt, so a deletion that fails keeps the secret counted.
    deletePassword: () => publishCredentialSync(() => entry.deletePassword(), { accounts: [account], absent }),
  };
}

function readKeychain(account: string, quiet = false): string | undefined {
  const cached = resolvedCache.get(account);
  if (cached !== undefined) return cached;
  let value: string | null = null;
  try {
    value = providerKeychainEntry(account).getPassword();
  } catch {
    // fall through to the single warning below
  }
  if (typeof value === "string" && value.trim()) {
    if (isGcpCredentialMaterial(value)) throw new GcpCredentialAsApiKeyError();
    resolvedCache.set(account, value);
    return value;
  }
  if (!quiet && !warnedAccounts.has(account)) {
    warnedAccounts.add(account);
    console.warn(`[opencodex] provider key reference keychain:${account} could not be read from the OS keychain; requests for this provider have no credential until the keychain is available (no plaintext fallback)`);
  }
  return undefined;
}

/**
 * Single resolver for provider key material: env references, keychain references, or the
 * literal value. Every request-time read of `apiKey` goes through here.
 */
export function resolveProviderApiKey(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const resolved = isKeychainReference(value) ? readKeychain(keychainAccount(value)) : resolveEnvValue(value);
  observeDecisionKeyCredential(value, resolved);
  return resolved;
}

/** The same resolution without the unreadable-reference warning, for observe-only callers. */
export function resolveProviderApiKeyQuietly(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const resolved = isKeychainReference(value) ? readKeychain(keychainAccount(value), true) : resolveEnvValue(value);
  // A quiet read still sees a replaced secret; advisory key evidence published under the old one must not outlive it.
  observeDecisionKeyCredential(value, resolved);
  return resolved;
}

export type ProviderKeyStoreKind = "keychain" | "env" | "file" | "none";

export function providerKeyStoreKind(provider: Pick<OcxProviderConfig, "apiKey"> | undefined): ProviderKeyStoreKind {
  const key = provider?.apiKey;
  if (!key) return "none";
  if (key.startsWith(GCP_CREDENTIAL_MARKER_PREFIX) || isKeychainReference(key)) return "keychain";
  if (/^\$\{?\w+\}?$/.test(key)) return "env";
  return "file";
}

/** Probe the OS keychain with a throwaway account: write, read back, delete. It stores no credential. */
export function probeProviderKeychain(): { available: true } | { available: false; reason: string } {
  const account = `probe-${process.pid}-${Date.now()}`;
  try {
    const entry = entryFactory(PROVIDER_KEYCHAIN_SERVICE, account);
    entry.setPassword("ok");
    const back = entry.getPassword();
    try { entry.deletePassword(); } catch { /* best effort */ }
    if (back !== "ok") return { available: false, reason: "keychain read-back did not match" };
    return { available: true };
  } catch (error) {
    return { available: false, reason: error instanceof Error ? error.message : "keychain unavailable" };
  }
}
