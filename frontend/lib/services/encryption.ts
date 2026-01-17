/**
 * Encryption service for securely storing sensitive data like API keys.
 *
 * Uses AES-256-GCM encryption with:
 * - 32-byte encryption key from environment variable
 * - Random 12-byte IV (initialization vector) for each encryption
 * - 16-byte authentication tag for integrity verification
 *
 * Format: base64(iv + authTag + ciphertext)
 *
 * IMPORTANT: This module should only be used server-side.
 * Never import this in client components.
 */

import crypto from 'crypto';

// Encryption configuration constants
const ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 12;        // GCM recommended IV length
const AUTH_TAG_LENGTH = 16;  // GCM auth tag length
const KEY_LENGTH = 32;       // 256 bits for AES-256

/**
 * Get the encryption key from environment variable.
 * Validates that the key exists and is the correct length.
 *
 * @throws Error if ENCRYPTION_KEY is not set or invalid
 */
function getEncryptionKey(): Buffer {
  const key = process.env.ENCRYPTION_KEY;

  if (!key) {
    throw new Error(
      'ENCRYPTION_KEY environment variable is not set. ' +
      'Generate one with: node -e "console.log(require(\'crypto\').randomBytes(32).toString(\'hex\'))"'
    );
  }

  // Convert hex string to buffer
  const keyBuffer = Buffer.from(key, 'hex');

  if (keyBuffer.length !== KEY_LENGTH) {
    throw new Error(
      `ENCRYPTION_KEY must be ${KEY_LENGTH * 2} hex characters (${KEY_LENGTH} bytes). ` +
      `Current length: ${key.length} characters.`
    );
  }

  return keyBuffer;
}

/**
 * Encrypt a plaintext string (e.g., API key).
 *
 * @param plaintext - The string to encrypt
 * @returns Base64-encoded encrypted data (iv + authTag + ciphertext)
 * @throws Error if encryption fails or key is invalid
 */
export function encryptApiKey(plaintext: string): string {
  const key = getEncryptionKey();

  // Generate random IV for this encryption
  const iv = crypto.randomBytes(IV_LENGTH);

  // Create cipher and encrypt
  const cipher = crypto.createCipheriv(ALGORITHM, key, iv);
  const encrypted = Buffer.concat([
    cipher.update(plaintext, 'utf8'),
    cipher.final()
  ]);

  // Get authentication tag for integrity verification
  const authTag = cipher.getAuthTag();

  // Combine: iv (12) + authTag (16) + ciphertext (variable)
  const combined = Buffer.concat([iv, authTag, encrypted]);

  return combined.toString('base64');
}

/**
 * Decrypt an encrypted API key.
 *
 * @param encryptedData - Base64-encoded encrypted data from encryptApiKey()
 * @returns The original plaintext API key
 * @throws Error if decryption fails, data is corrupted, or key is invalid
 */
export function decryptApiKey(encryptedData: string): string {
  const key = getEncryptionKey();

  // Decode from base64
  const combined = Buffer.from(encryptedData, 'base64');

  // Validate minimum length (iv + authTag + at least 1 byte of data)
  const minLength = IV_LENGTH + AUTH_TAG_LENGTH + 1;
  if (combined.length < minLength) {
    throw new Error('Encrypted data is too short or corrupted');
  }

  // Extract components
  const iv = combined.subarray(0, IV_LENGTH);
  const authTag = combined.subarray(IV_LENGTH, IV_LENGTH + AUTH_TAG_LENGTH);
  const ciphertext = combined.subarray(IV_LENGTH + AUTH_TAG_LENGTH);

  // Create decipher and decrypt
  const decipher = crypto.createDecipheriv(ALGORITHM, key, iv);
  decipher.setAuthTag(authTag);

  const decrypted = Buffer.concat([
    decipher.update(ciphertext),
    decipher.final()
  ]);

  return decrypted.toString('utf8');
}

/**
 * Generate a masked preview of an API key for display purposes.
 * Shows provider prefix and last 4 characters.
 *
 * Examples:
 * - "sk-ant-api03-abc...xyz123" -> "sk-ant-...z123"
 * - "sk-1234567890abcdef" -> "sk-...cdef"
 *
 * @param apiKey - The full API key
 * @returns A masked preview safe for display
 */
export function getKeyPreview(apiKey: string): string {
  if (!apiKey || apiKey.length < 8) {
    return '***';
  }

  // For Anthropic keys, show "sk-ant-" prefix pattern
  // For other keys, show first 3 chars
  const prefixMatch = apiKey.match(/^(sk-ant-|sk-)/);
  const prefix = prefixMatch ? prefixMatch[0] : apiKey.substring(0, 3);

  // Show last 4 characters
  const suffix = apiKey.slice(-4);

  return `${prefix}...${suffix}`;
}

/**
 * Validate that a string looks like a valid Anthropic API key.
 * Does NOT verify the key works - just checks format.
 *
 * @param apiKey - The API key to validate
 * @returns true if format appears valid
 */
export function isValidApiKeyFormat(apiKey: string): boolean {
  if (!apiKey || typeof apiKey !== 'string') {
    return false;
  }

  // Anthropic keys start with "sk-ant-" and are reasonably long
  // Minimum realistic length is around 40 characters
  const trimmed = apiKey.trim();
  return trimmed.startsWith('sk-ant-') && trimmed.length >= 40;
}
